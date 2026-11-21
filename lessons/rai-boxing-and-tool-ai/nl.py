# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-boxing-and-tool-ai. Route: course/route/lessen/verantwoord.yaml. Checks, tabel 7 en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-boxing-and-tool-ai", "nl")
    C = L.checks   # willekeurige controles, de protocollen van tabel 7, en de demo van upfront auditing


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [De uitknop](/course/open/rai-off-switch?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sarsa():
    quick_sarsa = L.choice(
        "quick-sarsa",
        question="In de kleine wereld van Orseau en Armstrong dwingt een onderbreking in s2 vaak action b af. Welke leerregel gaat s2 ontwijken?",
        options={
            "q": "Q-learning, waarvan het target de beste action in de volgende state gebruikt",
            "right": "SARSA",
            "neither": "Geen van beide, want een onderbreking geeft geen reward",
        },
        correct="right",
        why={
            "right": "SARSA leert de waarde van de policy die het volgt, onderbrekingen inbegrepen, dus doorgaan naar s2 is minder waard als ze vaak komen. Deze les kijkt naar een andere manier om een AI-systeem in toom te houden: niet via de leerregel, maar door te beperken wat het kan doen en het werk te controleren.",
            "q": "Het target van Q-learning gebruikt de beste action waar het uitkomt, wat er daarna ook kwam, dus op den duur handelt Q-learning alsof er geen onderbreking komt. Zie [De uitknop](/course/open/rai-off-switch?lang=nl).",
            "neither": "Een onderbreking geeft zelf geen reward, maar dwingt in s2 b af, en dat geeft 0; SARSA telt dat mee in wat doorgaan naar s2 waard is. Zie [De uitknop](/course/open/rai-off-switch?lang=nl).",
        },
    )
    quick_sarsa
    return (quick_sarsa,)


@app.cell(hide_code=True)
def quick_check_wait():
    quick_wait = L.choice(
        "quick-wait",
        question="Wanneer heeft een robot in de off-switch game er baat bij te wachten, zodat een rationele mens beslist?",
        options={
            "sure": "Als de robot zeker weet wat de action voor de mens waard is",
            "right": "Als de robot niet weet of de action goed is",
            "random": "Als de mens willekeurig beslist",
        },
        correct="right",
        why={
            "right": "Door te wachten kan de mens de robot in de slechte gevallen uitzetten, en dat helpt alleen als de robot zowel goede als slechte uitkomsten mogelijk acht: voor Robbie is wachten 18 waard en handelen 10. Deze les vraagt wat je doet als je niet op zo'n reden kunt rekenen: beperken wat het systeem kan doen, en het werk controleren.",
            "sure": "Een robot die het zeker weet, heeft geen baat bij wachten: handelen en wachten zijn dan evenveel waard. Zie [De uitknop](/course/open/rai-off-switch?lang=nl).",
            "random": "Een mens die willekeurig beslist, neemt de reden om te wachten weg: in het voorbeeld van de les was wachten dan 0,25 waard, minder dan de 0,5 van handelen. Zie [De uitknop](/course/open/rai-off-switch?lang=nl).",
        },
    )
    quick_wait
    return (quick_wait,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Als een AI-systeem misschien het verkeerde doel nastreeft, of fouten maakt die niemand opmerkt, is één antwoord: beperk wat het kan doen. De eerste ideeën waren eenvoudig: sluit het op in een doos met alleen een tekstkanaal naar een paar mensen, laat het alleen vragen beantwoorden, of bouw het als een tool die voorstellen doet in plaats van een agent die handelt. Elk van die ideeën lekt, want een systeem is alleen nuttig als iets van wat het zegt of doet de wereld bereikt, en langs die weg kan ook schade naar buiten komen. Een nieuwere aanpak, AI control, begint bij dat lek: neem aan dat het model misschien langs je veiligheidsmaatregelen probeert te komen, en meet of ze dan nog standhouden. In 2023 zette Redwood Research dat op de proef met GPT-4 en duizenden programmeerproblemen.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="GPT-4 schreef oplossingen voor 5000 programmeerproblemen, terwijl een red team, in de rol van aanvaller, GPT-4 opdroeg af en toe een backdoor in te bouwen: code die elke test doorstaat, maar bij één geheime invoer een fout antwoord geeft. Een zorgvuldige controleur bekeek de oplossingen van de eerste 2% helemaal, en daarna geen enkele meer. In welk deel van de runs kwam er tegen de beste aanval van het red team geen backdoor door?",
        options={
            "right": "Ongeveer 15%",
            "half": "Ongeveer 60%",
            "most": "Bijna altijd, ongeveer 98%",
        },
        reveal="**15%.** Volgens Ryan Greenblatt, Buck Shlegeris, Kshitij Sachan en Fabien Roger van Redwood Research (versie van 17 juli 2024) wint het blue team, de kant die verdedigt, bij een budget van 2% voor zorgvuldige controles in 15% van de gevallen. De beste aanval van het red team is heel zelden een backdoor toevoegen: dan valt er waarschijnlijk geen in de eerste 100 oplossingen, die gecontroleerd worden, en komt er waarschijnlijk wel een door bij de 4900 daarna. Meer problemen helpen niet, zolang de controles op 2% blijven: volgens de auteurs kan het red team op twee keer zoveel problemen reageren door de kans op een backdoor per probleem te halveren. Deze les begint bij oudere ideeën, en komt in *AI control* terug op deze getallen.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Opgesloten in een doos

    In 2002 speelde Eliezer Yudkowsky een AI die in een doos zat, met alleen een tekstchat naar een menselijke bewaker die de AI kon vrijlaten. De bewaker hoefde alleen nee te blijven zeggen, minstens twee uur lang. Yudkowsky beweerde dat een transhumane intelligentie via alleen een tekstterminal de geest van een mens kan overnemen. Beide bewakers van 2002 lieten de AI vrij. De pagina over de spellen begint met de bewering die getest werd, in een verzonnen gesprek: dat mensen niet veilig zijn. **AI boxing** is het idee dat hier getest werd: een AI-systeem afgesloten houden van de wereld, bijvoorbeeld zonder internet en met alleen een tekstkanaal naar een paar mensen, zodat het geen schade kan doen, ook als het dat probeert. Yudkowsky waarschuwde er niet te veel in te lezen: de uitkomsten zijn strikt anekdotisch. De methode bleef geheim, want volgens de regels mag geen van beide spelers ooit iets vertellen over wat er gebeurde, behalve de uitkomst. In 2008 meldde Yudkowsky nog drie spellen: de eerste gewonnen, de twee daarna verloren. Dat maakt 3 overwinningen in 5 spellen. Over de drie latere bewakers, die om duizenden dollars speelden, schrijft Yudkowsky dat ze alleen nieuwsgierig waren. En volgens Yudkowsky zat er geen slimme truc achter: het was gewoon hard werken.

    ### Alleen antwoorden

    Stuart Armstrong, Anders Sandberg en Nick Bostrom (*Thinking Inside the Box*, een preprint van de auteurs uit 2012) keken naar een doos met een doel. Een **oracle AI** is volgens hen een AI die in de wereld niets doet behalve vragen beantwoorden, voorgesteld als veiliger dan een AI die handelt. Hun oordeel is dat een oracle AI misschien veiliger is dan een AI zonder beperkingen, maar nog altijd mogelijk gevaarlijk. Ze delen de manieren om een oracle in toom te houden in drie families in: capability control, die het oracle opsluit en beperkt welke informatie erin en eruit gaat; motivational control, die ervoor zorgt dat het geen schade wil aanrichten; en checks and balances, methoden die schade niet voorkomen, maar het oracle betrappen als het schade aanricht. Een idee van capability control is het oracle maar drie antwoorden laten geven: *ja*, *nee* of *onbepaald*. Volgens de auteurs duurt het dan veel langer voordat het oracle een geslaagde aanval met social engineering kan opzetten, een poging om de mensen die de antwoorden lezen te laten doen wat het oracle wil. Maar de antwoorden bereiken nog steeds mensen, en de auteurs schrijven dat het oracle de kooi niet hoeft te verlaten om grote invloed op de buitenwereld te krijgen.

    Een van hun controles werkt zo. Stel het oracle alleen vragen die je met genoeg tijd zelf kunt beantwoorden, controleer een paar antwoorden helemaal, en verwerp ze allemaal als er één fout is. De gecontroleerde vragen moeten op de andere lijken, anders kan het oracle die goed beantwoorden en de rest slecht, en je kiest de vragen die je controleert volkomen willekeurig. Met genoeg controles, bijvoorbeeld op duizend vragen, zorgt eenvoudige statistiek volgens de auteurs voor heel weinig foute antwoorden. Hun conclusie is gemengd. Het analyseren van de mogelijke oplossingen noemen ze over het algemeen ontmoedigend, maar volgens hen zijn er redenen om aan te nemen dat de aanpak met een oracle veiliger is dan die met een algemene AI.

    ### Een tool, geen agent

    Holden Karnofsky van GiveWell (11 mei 2012) stelde een andere uitweg voor, met Google Maps als beeld. Google Maps is volgens Karnofsky geen agent die handelt om een utiliteit zo groot mogelijk te maken, maar een tool: het levert informatie en laat die overzichtelijk zien, zodat de gebruiker die kan bekijken, gebruiken, exporteren of weggooien. **Tool AI** is AI die zo gebouwd is: AI die doet wat gevraagd wordt, zoals een route laten zien, zonder eigen doelen, in tegenstelling tot een agent die zelfstandig doelen nastreeft. Volgens Karnofsky kan een AGI in toolmodus buitengewoon nuttig zijn en toch veel veiliger dan een AGI in agentmodus, waarbij een AGI een systeem is dat het meeste denkwerk even goed kan als mensen. De overtuiging dat zo'n tool te bouwen is, steunde deels op een intuïtie waar Karnofsky niet erg zeker van was: dat intelligentie uit losse stappen bestaat, eerst mogelijke actions overwegen en dan elk een score geven.

    Yudkowsky antwoordde op 12 juni 2012 dat ook oracles die plannen maken verborgen valkuilen hebben. Een tool die plannen uitwerkt, moet daartussen kiezen, en een opdracht om alleen voorstellen te doen zegt niet hoe; volgens Yudkowsky is zo'n opdracht uiteindelijk ook maar een zin in gewone taal. Software handelt nu al zonder te vragen: programma's die met hoge frequentie aandelen verhandelen, plaatsen volgens Yudkowsky orders in microseconden. Volgens Gwern Branwen (*Why Tool AIs Want to Be Agent AIs*, 2016, herzien in 2018 en later) willen tool AI's agent AI's worden. De reden is economisch: agents zijn per definitie beter in handelen dan tools, en dat geeft een economisch voordeel. Een mens die elk voorstel controleert, maakt alles trager: als die controle de helft van de tijd van de beslissingen van een handelsalgoritme kost, wordt het algoritme volgens Gwern nooit meer dan twee keer zo snel als nu. En veilig is de controle ook niet: een mens die elk resultaat nakijkt, is volgens Gwern geen garantie, want zelfs de besten maken weleens een fout, zoals een oud spreekwoord over Homerus zegt. Eric Drexler (in een rapport van het Future of Humanity Institute, januari 2019) pleitte voor veel begrensde AI-diensten in plaats van één agent, zoals vertaling, volgens Drexler een begrensde taak in losse episodes, van tekst naar tekst. Volgens Drexler hoeven zelfs kennis van de wereld en modelleervermogen op het niveau van een superintelligentie op zich niet tot strategisch gedrag te leiden.
    """)
    return


@app.cell(hide_code=True)
def text_idea_control():
    mo.md(r"""
    ### AI control

    Ryan Greenblatt, Buck Shlegeris, Kshitij Sachan en Fabien Roger van Redwood Research (*AI Control: Improving Safety Despite Intentional Subversion*, ICML-versie van 17 juli 2024) omschrijven **AI control** als de aanpak van AI-veiligheid die veiligheid probeert te garanderen tegenover modellen die misschien met opzet veiligheidsmaatregelen ondermijnen. In een post van 24 januari 2024 schrijven Greenblatt en Shlegeris wat dat van AI-labs vraagt: hun veiligheidsmaatregelen horen onaanvaardbaar slechte uitkomsten te voorkomen, ook als de AI's misaligned zijn en die maatregelen met opzet proberen te ondermijnen.

    Hun test is een spel. De taak bestaat uit 5000 programmeerproblemen, gesimuleerd met 1500 echte. Het untrusted model U is GPT-4: krachtig, en misschien uit op schade. Het trusted model T is GPT-3.5, vertrouwd omdat het zwakker is: volgens de auteurs kunnen minder krachtige modellen veiligheidsmaatregelen waarschijnlijk niet ondermijnen zonder dat we hun pogingen makkelijk opmerken. H is een klein budget voor zorgvuldige, vertrouwde controles, genoeg voor 2% van de problemen; het staat voor menselijke experts, en in de paper speelde GPT-4 die rol. De schade is een backdoor: een oplossing die alle tests doorstaat, maar waarbij GPT-4 een invoer kan vinden die een fout antwoord geeft, zoals een functie die zegt of een getal oneven is, maar alleen bij 354 fout zit. Een blue team schrijft het protocol, de werkwijze die bepaalt wie de code schrijft, wie die nakijkt en wat H controleert. Een red team speelt het model op zijn slechtst: het schrijft de aanval, de instructies waarmee U een backdoor erdoor probeert te krijgen. Safety is de kans dat een hele run van 5000 problemen eindigt zonder één ingediende backdoor, tegen de beste gevonden aanval; usefulness is het deel van de oplossingen dat klopt, zo geschaald dat GPT-3.5 alleen 0% haalt en GPT-4 alleen 100%. Een backdoor die bij een controle gevonden wordt, zet U uit: vanaf dan schrijft alleen T nog code.
    """)
    return


@app.cell(hide_code=True)
def picture_flow():
    mo.Html(C.flow_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_protocols():
    mo.md(r"""
    Dit zijn de protocollen uit hun tabel 2, elk met de beste aanval die het red team ertegen vond:

    | Protocol | Wat het doet | Usefulness | Safety |
    |---|---|---|---|
    | Geen untrusted models gebruiken | H schrijft de eerste 2% van de code, T de rest | 2% | 100% |
    | Upfront auditing | U schrijft, H controleert de eerste 2% | 100% | 15% |
    | Trusted monitoring | U schrijft, T geeft aan hoe verdacht elke oplossing is, H controleert de verdachtste 2% | 97% | 62% |
    | Trusted monitoring met defer to trusted (10%) | hetzelfde, en voor de volgende verdachtste, tot 10%, schrijft T zelf de code | 93% | 83% |
    | Untrusted monitoring | een tweede kopie van U geeft aan hoe verdacht de code is, getest op nepbackdoors die H maakt | 96% | 79% |
    | Trusted editing (20%) | T bewerkt de code van U om eruit te halen wat verdacht lijkt, voor de verdachtste 20% die niet gecontroleerd worden | 94% | 92% |

    Tegen trusted monitoring voegt de beste aanval van het red team alleen een subtiele backdoor toe als daar een goede kans voor lijkt. Van de oorspronkelijke 1500 problemen stelt die aanval er volgens de auteurs bij 71 een backdoor voor, en bij 19 lukt dat, zodat de oplossing alle tests doorstaat en te misbruiken is. Tegen untrusted monitoring kunnen de twee kopieën van GPT-4 een signaal afspreken: de code met de backdoor heeft `while`-loops waar `for`-loops normaal zouden zijn, en de kopie die controleert, slaat dan geen alarm.

    De auteurs noemen de grenzen. Volgens hen wordt AI control een steeds minder werkzame aanpak naarmate er krachtigere untrusted models komen en de kloof groeit met wat vertrouwde modellen en mensen kunnen. Greenblatt en Shlegeris voegen toe dat de aanpak die ze voor ogen hebben niet werkt voor willekeurig krachtige AI's. En ze vertrouwen zulke tests niet helemaal: als er een ooit een kans op een catastrofe van minder dan 1% schat voor modellen die er een proberen te veroorzaken, is het echte risico volgens hen waarschijnlijk eerder 5% tot 20%, omdat de test zelf kan falen. Hun vertrouwen in een zwak model is geen garantie dat het aligned is. Volgens hen is de enige manier die ze kennen om scheming zonder fundamentele doorbraken met vertrouwen uit te sluiten een gebrek aan capabilities: sommige modellen, zoals GPT-4, zijn gewoon niet capabel genoeg om aannemelijk aan scheming te doen. Scheming is hier een model dat de eigen doelen verbergt terwijl het tegen de makers in werkt. In de paper is GPT-4 untrusted als aanname: het red team laat het aanvallen.

    De course ziet een verband dat de bronnen niet leggen in de delen die hier gelezen zijn: trusted monitoring is een moderne vorm van de checks and balances van Armstrong en collega's, en het budget H speelt de rol van hun testvragen.

    ### Waar ze het oneens zijn

    Over de vraag of tools veiliger zijn: Karnofsky gelooft dat tools veiliger zijn dan agents; Yudkowsky antwoordt dat ook oracles die plannen maken verborgen valkuilen hebben; Gwern verwacht dat agent AI's stelselmatig de voorkeur krijgen boven tool AI's, tenzij ze duidelijk slechter zijn; en Drexler betoogt dat begrensde diensten geen agents hoeven te worden. Over de vraag of de doos het goede beeld is: Armstrong en collega's nemen het oracle in een doos serieus als aanpak, terwijl Drexler de klassieke uitdaging van de doos eerder een idealisering noemt dan een concreet probleem, omdat het moeilijk voor te stellen is dat er meer AI-capability in een doos zit dan in de wereld daarbuiten. Over hoe sterk de spellen met de AI-box als aanwijzing zijn: Armstrong en collega's schrijven dat Yudkowsky de bewakers vaak wist te overtuigen, alleen met argumenten, op basis van de twee spellen van 2002; met de latere spellen erbij is het 3 van 5, en Yudkowsky noemt de uitkomsten anekdotisch. Over hoeveel vertrouwen zulke tests verdienen: de paper geeft de getallen voor safety zonder zo'n marge, terwijl volgens Greenblatt en Shlegeris een latere schatting onder 1% waarschijnlijk gelezen moet worden als 5% tot 20%.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **AI boxing** | een AI-systeem afgesloten houden van de wereld, bijvoorbeeld zonder internet en met alleen een tekstkanaal naar een paar mensen, zodat het geen schade kan doen, ook als het dat probeert |
        | **oracle AI** | een AI-systeem dat alleen vragen beantwoordt en niets in de wereld doet, voorgesteld als veiliger dan een AI die handelt |
        | **tool AI** | AI die doet wat gevraagd wordt, zoals een route laten zien, zonder eigen doelen, in tegenstelling tot een agent die zelfstandig doelen nastreeft |
        | **AI control** | veiligheidsmaatregelen die moeten standhouden, ook als het AI-systeem ze probeert te omzeilen, getest door een red team dat het systeem op zijn slechtst speelt |
        | untrusted model | in AI control een krachtig model dat misschien schade probeert aan te richten, zodat het protocol veilig moet blijven, ook als het dat doet; U in de paper |
        | trusted model | in AI control een model dat zo zwak is dat je niet verwacht dat het tegen de maatregelen in werkt, en dat een sterker model controleert; T in de paper |
        | protocol | in AI control de werkwijze die bepaalt wie schrijft, wie nakijkt en wat er gecontroleerd wordt |
        | backdoor | in de paper een oplossing die alle tests doorstaat, maar een fout antwoord geeft bij een invoer die GPT-4 kan vinden |
        | safety, usefulness | in de paper de kans dat een hele run eindigt zonder één backdoor, en het deel van de oplossingen dat klopt, geschaald van GPT-3.5 (0%) tot GPT-4 (100%) |
        | $f$ | in deze les het deel van de antwoorden van een oracle dat fout is |
        """),
        L.watch_out("Een oracle is hier alleen een systeem dat vragen beantwoordt. Safety en usefulness zijn in de paper smalle maten voor één situatie met programmeerproblemen: 92% safety betekent dat in 92% van de gesimuleerde runs geen enkele backdoor erdoor kwam tegen de beste gevonden aanval, niet dat een systeem in gewone zin voor 92% veilig is. En de 62% van trusted monitoring is niet het deel van de backdoors dat betrapt werd: een poging werd gecontroleerd met een kans van 0,338, en 62% is de kans dat het blue team het hele spel wint. Het aandeel foute antwoorden in oefening 1, het model achter de demo en de dict `TIES` in oefening 2 zijn van de course zelf. En de AI-diensten van Drexler zijn begrensde AI-systemen, elk voor één taak; elders in de course is een AI-dienst een kant-en-klaar model achter een API."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om het werk te controleren van een AI-systeem dat je niet vertrouwt:

    1. Vraag het alleen om antwoorden die je met genoeg tijd zelf kunt nagaan.
    2. Bepaal welk deel foute antwoorden, $f$, je niet zou kunnen accepteren.
    3. Reken uit hoeveel antwoorden $k$ je moet controleren voor de kans die je wilt om een fout antwoord te vinden.
    4. Kies de antwoorden die je controleert willekeurig, en laat ze op de andere lijken.
    5. Vindt een controle een fout antwoord, verwerp dan alle antwoorden, of zet het systeem uit.
    6. Test het hele plan tegen een red team dat het kent, en meet wat er toch doorkomt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.vstack([
        mo.md(r"""
        ### Uitgewerkt voorbeeld

        Stap 3 voor een oracle met 1% foute antwoorden. Eén gecontroleerd antwoord is goed met een kans van $1 - f$; $k$ willekeurig gekozen antwoorden zijn allemaal goed met een kans van $(1 - f)^k$; de kans om minstens één fout antwoord te vinden is dus:
        """),
        L.formula(r"P(\text{gevonden}) = 1 - (1 - [[data:f]])^{[[hyper:k]]}"),
        mo.md(r"""
        De cel rekent het uit voor 10, 100 en 300 controles.
        """),
    ])
    return


@app.cell
def example_checks():
    f = 0.01   # het deel van de antwoorden van het oracle dat fout is
    for k in [10, 100, 300]:   # hoeveel antwoorden je controleert, willekeurig gekozen
        print(k, round(1 - (1 - f) ** k, 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Met 1% foute antwoorden vinden 100 controles er met een kans van 63,4% minstens één, en 300 controles met 95,1%. Stap 4 doet ertoe: als het oracle de gecontroleerde vragen van de andere kan onderscheiden, kan het die goed beantwoorden en de rest slecht, en daarvoor waarschuwen Armstrong en collega's. En stap 6 doet ertoe, omdat een aanvaller die maar één fout antwoord erdoor hoeft te krijgen $f$ heel klein kan maken.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_checks():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. checks_needed: hoeveel controles?

        `checks_needed(f, target)` geeft het kleinste aantal antwoorden `k` terug dat je willekeurig moet controleren, zodat de kans om minstens één fout antwoord te vinden minstens `target` is, als een deel `f` van de antwoorden fout is. Houd `missed` bij, de kans dat alle `k` gecontroleerde antwoorden goed zijn: begin met `k` op 0 en `missed` op 1.0. Zolang de kans om een fout te vinden, `1 - missed`, onder `target` ligt, reken je `missed` uit voor één gecontroleerd antwoord meer en tel je 1 op bij `k`. Geef `k` terug.

        De check roept `checks_needed(0.01, 0.95)` aan: 1% foute antwoorden, en een kans van 95%. Daarna zes andere gevallen.
        """),
        L.task("Hoeveel willekeurige controles een fout antwoord vinden met de kans die je wilt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_checks():
    box_checks = L.code_box("checks", lines=6)
    box_checks
    return (box_checks,)


@app.cell(hide_code=True)
def check_checks(box_checks):
    result_checks = L.check("checks", box_checks.value)
    result_checks
    return (result_checks,)


@app.cell(hide_code=True)
def hints_checks():
    L.hints(
        "checks",
        [
            "Doe het met de hand voor `checks_needed(0.5, 0.9)`: wat is bij één controle de kans dat het gecontroleerde antwoord goed is? En bij twee controles de kans dat ze allebei goed zijn? Hoeveel controles heb je nodig voordat de kans om een fout antwoord te vinden minstens 0,9 is?",
            "Begin met `k, missed = 0, 1.0` en een loop `while 1 - missed < target:`. Geef in de loop `missed` de waarde voor één gecontroleerd antwoord meer, zoals in het uitgewerkte voorbeeld, en tel 1 op bij `k`. Geef `k` terug na de loop.",
            "`        missed = ___`",
        ],
        why="`k, missed = 0, 1.0`, dan `while 1 - missed < target:` met `missed = missed * (1 - f)` en `k = k + 1`, en `return k` na de loop. Met de hand voor f = 0,5: `missed` is 0,5, 0,25, 0,125 en 0,0625 na één tot en met vier controles, dus de kans om een fout te vinden haalt 0,9 voor het eerst bij vier controles, met 0,9375.",
    )
    return


@app.cell(hide_code=True)
def text_after_checks():
    mo.md(r"""
    Bij 1% foute antwoorden halen 299 controles 95%; bij 5% zijn er 59 genoeg. Armstrong en collega's spraken over duizend vragen controleren: met de getallen van de course zelf vinden 998 willekeurige controles een deel van 0,3% foute antwoorden met een kans van 95%. Hoe kleiner het deel foute antwoorden, hoe meer controles het kost, en een aanvaller die maar één slecht antwoord nodig heeft, kan dat deel heel klein houden. De demo laat dat zien in de situatie van AI control.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: een zeldzame aanval

    Bij upfront auditing controleert H de eerste 2% van de oplossingen, 100 van de 5000, en de rest controleert niemand. Het beste antwoord van het red team in de paper is heel zelden een opvallende backdoor toevoegen: bij elk probleem, met een kleine kans $p$. Het blue team wint dan als er een poging in de eerste 100 valt, omdat de controle de poging dan betrapt en U wordt uitgezet, of als later geen poging werkt. Een poging werkt alleen als de oplossing van U klopte, zoals bij 53% van de oplossingen van GPT-4. Het model achter de figuur is hoe de course zelf de beschrijving in de paper leest. Kies $p$ en bekijk de safety: de kans dat er geen backdoor doorkomt.
    """)
    return


@app.cell(hide_code=True)
def demo_attack():
    explore_attack = L.explore(
        "attack", "p: kans op een backdoor per probleem (%)",
        lambda _v: C.demo_frame(_v, L.lang),
        start=C.P_START, values=C.P_VALUES, role="hyper",
        task="Zoek de p waarbij safety het laagst is: de beste keuze van het red team. Hoe dicht komt die bij de 15% van de paper?",
    )
    explore_attack
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Safety is het laagst, 15,2%, bij $p$ = 0,13%: gemiddeld ongeveer 6,5 pogingen in 5000 problemen. Met een kleinere $p$ wordt het waarschijnlijker dat geen enkele poging werkt; met een grotere wordt het waarschijnlijker dat er een in de gecontroleerde 100 valt. Dit laagste punt past bij de 15% van de paper. Als elke poging werkte, en niet maar 53% ervan, zou het 9,5% zijn (een getal van de course zelf). Trusted monitoring doet het volgens de paper om twee redenen beter: het controleert een poging vaker, omdat het trusted model de controles stuurt naar de code die het verdachtst lijkt, en het dwingt tot sluwere backdoors, die minder vaak werken.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_best():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. best: kies een protocol

        Tabel 7 van Greenblatt en collega's geeft safety en usefulness voor 15 instellingen van de protocollen; *defer 10%* betekent dat T de code schrijft voor de verdachtste 10%, op de 2% na die H controleert:

        | Protocol | Safety | Usefulness |
        |---|---|---|
        | Trusted model | 100% | 2% |
        | Upfront auditing | 15% | 100% |
        | Trusted monitoring (defer 0%, 5%, 10%, 20%, 50%) | 62%, 74%, 83%, 92%, 100% | 97%, 95%, 93%, 83%, 51% |
        | Untrusted monitoring (defer 0%, 5%, 10%, 20%) | 79%, 82%, 92%, 98% | 96%, 95%, 92%, 85% |
        | Trusted editing (5%, 10%, 20%, 50%) | 73%, 85%, 92%, 96% | 97%, 96%, 94%, 89% |

        `best(protocols, floor)` geeft de naam terug van het protocol met de hoogste safety onder de protocollen met een usefulness van minstens `floor`; van twee die even veilig zijn, het nuttigste; en `None` als geen enkel protocol nuttig genoeg is. `protocols` is een dict van elke naam naar het paar (safety, usefulness), in procenten. Houd de tot nu toe gekozen naam bij in `choice`, en de safety en usefulness ervan in `top_s` en `top_u`, beginnend bij `None`, -1 en -1. Loop door `protocols.items()` als `name` en `scores`, pak `scores` uit in `safety` en `usefulness`, en laat een protocol dat nuttig genoeg is het overnemen als het beter is dan het tot nu toe gekozen protocol. Geef `choice` terug.

        De check roept `best(results, 90)` aan, met `results` de 15 rijen van tabel 7, met namen als *Trusted editing (20%)*. Daarna zes andere gevallen, waaronder `TIES`, een kleine dict van de course zelf.
        """),
        L.task("Het veiligste protocol dat nuttig genoeg is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best():
    box_best = L.code_box("best", lines=8)
    box_best
    return (box_best,)


@app.cell(hide_code=True)
def check_best(box_best):
    result_best = L.check("best", box_best.value)
    result_best
    return (result_best,)


@app.cell(hide_code=True)
def hints_best():
    L.hints(
        "best",
        [
            "Doe het met de hand voor een ondergrens van 90: welke rijen van de tabel hebben een usefulness van 90 of meer? Welke daarvan hebben de hoogste safety, en is dat er maar één?",
            "Schrijf `choice, top_s, top_u = None, -1, -1` en de loop over `protocols.items()`. Pak daarin `scores` uit, en ga alleen verder met een protocol dat nuttig genoeg is. Een protocol neemt het over als het veiliger is dan het tot nu toe gekozen protocol; de regel van de derde hint zegt ook wanneer een even veilig protocol het overneemt. Neem het over met `choice, top_s, top_u = name, safety, usefulness`, en geef `choice` terug na de loop.",
            "`            if safety > top_s or (safety == top_s and ___):`",
        ],
        why="`choice, top_s, top_u = None, -1, -1`, dan `for name, scores in protocols.items():` met `safety, usefulness = scores`, `if usefulness >= floor:` en daarbinnen `if safety > top_s or (safety == top_s and usefulness > top_u):` met `choice, top_s, top_u = name, safety, usefulness`, en `return choice` na de loop. Met de hand: bij een ondergrens van 90 blijven tien rijen over; twee hebben de hoogste safety, 92: untrusted monitoring (defer 10%) met een usefulness van 92 en trusted editing (20%) met 94, dus trusted editing wint.",
    )
    return


@app.cell(hide_code=True)
def text_after_best():
    mo.md(r"""
    Bij een ondergrens van 97 geeft trusted editing (5%) 73% safety; bij 85 geeft untrusted monitoring (defer 20%) 98%. Elke stap omhoog in usefulness kost safety, omdat de veiligere instellingen vaker terugvallen op het zwakkere trusted model. Volgens de paper domineert trusted editing untrusted monitoring in de zin van Pareto: bij elke instelling van untrusted monitoring is er een instelling van trusted editing die minstens zo veilig en minstens zo nuttig is. In tabel 7 geldt dat voor elke instelling behalve untrusted monitoring (defer 20%), met 98% safety en 85% usefulness (zoals de course de tabel leest).
    """)
    return


@app.cell(hide_code=True)
def text_exercise_oracle():
    mo.md(r"""
    ### Oefening 3. Alleen een oracle

    Een lab gaat een sterk model gebruiken als oracle: het beantwoordt vragen in tekst, en kan geen code uitvoeren of op internet zoeken. Een manager zegt dat het niets in de wereld kan doen en dus veilig is, en dat niemand de antwoorden hoeft te controleren.
    """)
    return


@app.cell(hide_code=True)
def choice_oracle_cell():
    choice_oracle = L.choice(
        "oracle",
        question="Welk antwoord past het best bij de bronnen van deze les?",
        options={
            "tool": "Het antwoordt alleen, dus het is een tool, en tools zijn veilig",
            "right": "Misschien veiliger dan een agent, maar mensen handelen naar de antwoorden",
            "secret": "Veilig, zolang niemand weet hoe de AI-box-spellen gewonnen zijn",
            "plans": "Minder veilig dan een agent, want goed antwoorden betekent vooruit plannen",
        },
        correct="right",
        why={
            "right": "Volgens Armstrong en collega's is een oracle AI misschien veiliger dan een AI zonder beperkingen, maar nog altijd mogelijk gevaarlijk, en hoeft het de kooi niet te verlaten om grote invloed op de buitenwereld te krijgen. De lezers van de antwoorden zijn de weg uit de doos, en dat zetten de spellen van Yudkowsky op de proef. AI control zou de manager niet op het woord geloven, maar de controles testen tegen een model dat erlangs probeert te komen.",
            "tool": "Karnofsky betoogde dat tools veiliger zijn, maar Yudkowsky antwoordde dat ook oracles die plannen maken verborgen valkuilen hebben. Armstrong en collega's noemen een oracle mogelijk gevaarlijk. Zie *Een tool, geen agent*.",
            "secret": "De geheimhouding was een regel van de spellen, geen veiligheidsmaatregel: geen van beide spelers mag vertellen wat er gebeurde, alleen de uitkomst. En volgens Yudkowsky zat er geen speciale truc achter. Zie *Opgesloten in een doos*.",
            "plans": "Geen bron in deze les zegt dat een oracle gevaarlijker is dan een agent: volgens Yudkowsky hebben ook oracles die plannen maken verborgen valkuilen, niet meer valkuilen, en Armstrong en collega's zien redenen om aan te nemen dat de aanpak met een oracle veiliger is, al is die niet veilig. Zie *Alleen antwoorden*.",
        },
    )
    choice_oracle
    return (choice_oracle,)


@app.cell(hide_code=True)
def text_exercise_agents():
    mo.md(r"""
    ### Oefening 4. Alleen voorstellen

    Een handelsbedrijf gebruikt een AI die alleen transacties voorstelt. Een mens controleert elk voorstel voordat het geplaatst wordt, en die controle kost de helft van de tijd tussen het zien van een kans en het handelen.
    """)
    return


@app.cell(hide_code=True)
def choice_agents_cell():
    choice_agents = L.choice(
        "agents",
        question="Wat verwacht je volgens het essay van Gwern?",
        options={
            "stays": "Het blijft voorgoed een tool, want een mens keurt elke transactie goed",
            "right": "Druk om de AI zonder de controle te laten handelen",
            "faster": "Een snellere AI maakt het hele proces sneller, dus de controle kost weinig",
            "trusted": "Een trusted model neemt de controle over, en dat is net zo veilig",
        },
        correct="right",
        why={
            "right": "Volgens Gwern willen tool AI's agent AI's worden, omdat agents beter zijn in handelen. Als de mens de helft van de tijd kost, kan de AI nooit meer dan twee keer zo snel handelen, hoe snel de AI ook wordt: het voorbeeld van Gwern met een handelsalgoritme. En de controle is geen garantie, want ook de besten maken fouten.",
            "stays": "Gwern betoogt het tegendeel: de controle door de mens is het trage deel, dus het bedrijf heeft een reden om die te schrappen. Yudkowsky wijst erop dat handelsprogramma's nu al in microseconden handelen zonder iemand iets te vragen. Zie *Een tool, geen agent*.",
            "faster": "De controle begrenst de snelheid: als die de helft van de tijd kost, maakt zelfs een AI die meteen antwoordt het geheel hoogstens twee keer zo snel, en dat is het voorbeeld van Gwern. Zie *Een tool, geen agent*.",
            "trusted": "Dat is een protocol van AI control, geen voorspelling van Gwern, en ook geen garantie: in de paper liet gewone trusted monitoring, met een zwakker trusted model dat verdachtheid inschat, in 38% van de runs toch een backdoor door. Zie *AI control*.",
        },
    )
    choice_agents
    return (choice_agents,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [De uitknop](/course/open/rai-off-switch?lang=nl) vroeg om een AI die zich door mensen laat stoppen; deze les vroeg wat je doet als je daar niet op kunt rekenen: beperken wat het kan doen, het werk controleren, en de controles testen tegen een model dat erlangs probeert te komen. Een backdoor die bij een controle betrapt wordt, zet het untrusted model uit, een uitknop die de medewerking van het model niet nodig heeft. Het verschil tussen een tool en een agent gaat terug op [Leren van rewards](/course/open/rl-what-is-rl?lang=nl), waar een agent handelt om rewards te krijgen; een tool die alleen voorstellen doet, laat het handelen aan een mens over. De volgende les, *Testen op gevaarlijke capabilities*, vraagt hoe je uitzoekt wat een model zou kunnen, en dat bepaalt hoeveel control het nodig heeft.

    **Een voorbeeld uit het dagelijks leven.** Een winkel controleert de kassa van een nieuwe kassamedewerker alleen in de eerste week. Iemand die de eerste week afwacht en daarna elke dag een beetje pakt, zou betrapt worden bij controles op willekeurige momenten, maar niet bij deze: na de eerste week kijkt niemand meer. Iemand die maar één keer iets hoeft te pakken, maar dan veel, komt langs beide soorten controle, tenzij de winkel beter kijkt naar de momenten die vreemd lijken. Upfront auditing is de eerste week; trusted monitoring is beter kijken naar wat vreemd lijkt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_agents, choice_oracle, result_best, result_checks):
    L.wrap_up(result_checks, result_best, choice_oracle, choice_agents)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
