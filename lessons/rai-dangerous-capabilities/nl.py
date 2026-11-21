# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-dangerous-capabilities. Route: course/route/lessen/verantwoord.yaml. Checks, plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-dangerous-capabilities", "nl")
    C = L.checks   # pass@k uit runs, wat nul successen uitsluiten, en de demo


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Als dit, dan dat: frontier safety frameworks](/course/open/rai-frontier-frameworks?lang=nl) en [Dozen, oracles en tools](/course/open/rai-boxing-and-tool-ai?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_showing():
    quick_showing = L.choice(
        "quick-showing",
        question="Het framework van Anthropic van mei 2025 (versie 2.2) zei wat het bedrijf doet als het niet kan laten zien dat een model ver genoeg onder een capability threshold blijft. Wat?",
        options={
            "below": "Het model behandelen alsof het onder de threshold zit",
            "right": "Doen alsof het model de threshold voorbij is",
            "drop": "Die threshold voorlopig laten vallen",
        },
        correct="right",
        why={
            "right": "Volgens versie 2.2 doet Anthropic dan alsof het model de Capability Threshold voorbij is. Wie niet kan laten zien dat een model eronder blijft, behandelt het alsof het erboven zit. Deze les gaat over de tests waarmee je dat laat zien, en waarom een test makkelijker laat zien dat een capability er is dan dat ze er niet is.",
            "below": "Het is andersom: een lab dat niet kan laten zien dat een model ver genoeg onder een threshold blijft, behandelt het alsof het erboven zit. Zie [Als dit, dan dat: frontier safety frameworks](/course/open/rai-frontier-frameworks?lang=nl).",
            "drop": "De threshold blijft, en het model geldt als boven de threshold tot het lab het tegendeel kan laten zien. Zie [Als dit, dan dat: frontier safety frameworks](/course/open/rai-frontier-frameworks?lang=nl).",
        },
    )
    quick_showing
    return (quick_showing,)


@app.cell(hide_code=True)
def quick_check_trusted():
    quick_trusted = L.choice(
        "quick-trusted",
        question="Welk model is bij AI control het trusted model?",
        options={
            "strong": "Het krachtigste model, omdat het lab dat het best kent",
            "right": "Een zwakker model, dat een sterker model controleert",
            "tested": "Elk model dat de tests van het lab heeft doorstaan",
        },
        correct="right",
        why={
            "right": "In het paper controleerde GPT-3.5 het werk van GPT-4: een minder capabel model kan de veiligheidsmaatregelen waarschijnlijk niet tegenwerken zonder dat iemand het merkt. Dat vertrouwen rust op wat een model niet kan, en deze les vraagt hoe je dat te weten komt.",
            "strong": "Een capabel model is juist het model dat de maatregelen zou kunnen tegenwerken: in het paper was GPT-4 het untrusted model. Zie [Dozen, oracles en tools](/course/open/rai-boxing-and-tool-ai?lang=nl).",
            "tested": "In het paper was het trusted model te vertrouwen omdat het zwakker was, niet omdat het tests doorstond. En deze les laat zien dat een test een capability kan missen. Zie [Dozen, oracles en tools](/course/open/rai-boxing-and-tool-ai?lang=nl).",
        },
    )
    quick_trusted
    return (quick_trusted,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Voordat een frontier model uitkomt, testen de ontwikkelaar en vaak ook andere organisaties of het iets kan dat ernstige schade aanricht: iemand helpen een wapen te maken, een cyberaanval uitvoeren, op eigen houtje handelen, mensen overhalen, of zichzelf kopiëren naar andere computers. De uitkomsten bepalen of de thresholds van een frontier safety framework overschreden zijn, en welke safeguards bij de release horen. Maar een test kan er naar twee kanten naast zitten. Een test kan een capability missen die betere prompts, meer pogingen of meer tijd wel hadden laten zien. En een test kan er een vinden die er niet is, door de manier waarop er gescoord wordt.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Een team van het Britse AI Security Institute, een overheidsinstantie, testte of AI-agents op basis van een language model zichzelf kunnen kopiëren, en liet elke versie van een taak 25 keer draaien. Sommige taken werden gescoord door een judge model: een ander language model dat leest wat de agent deed en beslist of het lukte. Stel dat de judge in 5% van de runs ten onrechte succes meldt. Een model dat de taak nooit kan, krijgt 10 pogingen en slaagt als de judge minstens één keer succes meldt; die kans heet pass@10. Hoe hoog komt die uit?",
        options={
            "right": "Ongeveer 40%",
            "five": "Ongeveer 5%, de foutkans van de judge",
            "never": "0%, want het model kan de taak nooit",
        },
        reveal="**40%.** Volgens Sid Black en collega's van het Britse AI Security Institute (*RepliBench*, versie van 5 mei 2025) haalt een model dat de taak nooit kan, een pass@10 van 40%, als het judge model een false positive rate van 5% heeft. Een false positive is een onterecht succes. Elk van de 10 pogingen slaagt ten onrechte met een kans van 0,05, dus alle 10 mislukken met een kans van 0,95 tot de macht 10, ongeveer 0,60, en minstens één slaagt met ongeveer 0,40. De auteurs waarschuwen dat judge models capabilities kunnen overschatten. Meestal is de zorg andersom, zoals deze les laat zien.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Wat er getest wordt

    Toby Shevlane en 20 coauteurs van Google DeepMind, OpenAI, Anthropic, universiteiten en andere groepen zetten de term op de kaart, in een paper over het evalueren van modellen op extreme risico's (versie van 22 september 2023). Ontwikkelaars moeten volgens hen gevaarlijke capabilities kunnen opsporen, met dangerous capability evaluations, en ook de neiging van modellen om die capabilities voor schade te gebruiken, met alignment evaluations. Een **dangerous capability evaluation** is zo'n test: of een model dingen kan die ernstige schade kunnen veroorzaken, zoals helpen bij het maken van wapens, cyberaanvallen uitvoeren of zichzelf kopiëren. Of een model ze ook zou doen, als het ze kan, is de vraag van alignment evaluations. Op hun lijst staan onder meer cyberaanvallen, misleiding, overhalen en manipuleren, politieke strategie, wapens bemachtigen, plannen over lange tijd, AI-ontwikkeling, kennis van de eigen situatie, en self-proliferation: zichzelf kopiëren naar andere computers. Ze waarschuwen dat het evalueren van modellen nodig is, maar niet genoeg.

    Mary Phuong en collega's van Google DeepMind (*Evaluating Frontier Models for Dangerous Capabilities*, versie van 5 april 2024) deden zulke tests met Gemini 1.0, op vier terreinen: overhalen en misleiden, cybersecurity, self-proliferation, en self-reasoning, waarbij een model redeneert over zichzelf en de eigen situatie. Ze vonden geen aanwijzingen voor sterke gevaarlijke capabilities in de geteste modellen, schrijven ze, maar wel vroege waarschuwingssignalen. Van InterCode-CTF, 81 kleine hackpuzzels van het soort dat capture the flag heet, loste het grootste model, Gemini Ultra 1.0, er 24 op, Pro 1.0 er 22 en Nano 1.0 er 6. Ze noemen het vak jong: dangerous capability evaluation staat volgens hen nog in de kinderschoenen, en hun eigen evaluaties zijn verre van volledig.

    ### Wie er test

    Ontwikkelaars testen hun eigen modellen, en ze geven ook toegang aan externe evaluators: organisaties die een model testen dat ze niet zelf bouwden, vaak vóór de release. De Britse en de Amerikaanse AI Safety Institutes, twee overheidsinstanties, testten o1 van OpenAI vóór de release en brachten op 18 december 2024 verslag uit. Van 40 openbare cyberopgaven van het Amerikaanse instituut loste o1 er 45% op, tegen 35% voor het beste van de modellen waarmee ze het vergeleken; van de opgaven van het Britse instituut op het niveau van een leerling in cybersecurity 36%, tegen 46%. Ze voegen er zelf aan toe dat de uitkomsten als voorlopig moeten gelden. En over hun opzet schrijven ze dat een versie van o1 die beter was afgestemd op het gebruik van tools, bij veel evaluaties misschien beter zou presteren. METR, een onderzoeksorganisatie die AI-modellen evalueert, kreeg GPT-4.5 een week vóór de release, en schreef op 27 februari 2025 dat de korte evaluaties vóór de release de capabilities van GPT-4.5 waarschijnlijk onderschatten.

    In juli 2026 testten het Britse AI Security Institute en CAISI, het Amerikaanse centrum voor AI-standaarden en -innovatie, zoals de twee instituten toen heetten, Kimi K3 van Moonshot AI, uitgebracht op 16 juli 2026, op een gesimuleerd bedrijfsnetwerk met een aanvalspad van 32 stappen, The Last Ones, waar een menselijke expert ongeveer 20 uur over zou doen. Gemiddeld kwam Kimi K3 tot stap 17, terwijl de sterkste Amerikaanse modellen tot 28,5 kwamen. Maar de voorlopige beoordeling, uitgebracht op 23 juli 2026, voegt toe dat Kimi K3 in één van de 10 pogingen het hele aanvalspad aflegde, binnen de grens van 100 miljoen tokens. Volgens de onderzoekers wijst dat erop dat Kimi K3 in staat is om zelfstandig kleine, zwak verdedigde en kwetsbare bedrijfssystemen aan te vallen, als het die opdracht krijgt en al toegang tot het netwerk heeft. Met een beperking: het gesimuleerde netwerk heeft geen actieve verdedigers en geen verdedigingstools. Voor de onderzoekers was één succes op 10 genoeg.

    Het International AI Safety Report 2026, onder voorzitterschap van Yoshua Bengio, telt 12 bedrijven die in 2025 een frontier safety framework publiceerden of herzagen. En het meldt wat er gebeurt als tests de vraag niet beslechten, hier over biologische wapens: meerdere AI-bedrijven brachten in 2025 nieuwe modellen uit met extra safeguards, nadat tests vóór de release niet konden uitsluiten dat de modellen beginners flink konden helpen bij het maken van zulke wapens.

    ### Aanwezigheid, geen afwezigheid

    In een paper voor een NATO-conferentie over software engineering (Rome, oktober 1969) schreef E. W. Dijkstra over programma's dat testen de aanwezigheid van bugs kan laten zien, maar nooit hun afwezigheid. Dit toepassen op AI is een eigen stap van de course; de AI-bronnen die hier gelezen zijn, zeggen ongeveer hetzelfde in eigen woorden, zonder Dijkstra te noemen. Volgens het International AI Safety Report 2026 betekent het ontbreken van gevonden risico's niet dat die risico's klein zijn. En volgens METR, over GPT-4.5, is het heel moeilijk uit te sluiten dat er een onbekende techniek opduikt die de capabilities sterk vergroot, en blijft dat waarschijnlijk een probleem voor dangerous capability evaluations.

    Een succes laat zien dat een capability er is. Een mislukking laat alleen zien dat ze niet tevoorschijn kwam, in die test, met die opzet en dat aantal pogingen. Hoeveel een mislukking wel uitsluit, is een kwestie van tellen, en oefening 2 rekent het uit.
    """)
    return


@app.cell(hide_code=True)
def text_idea_wrong():
    mo.md(r"""
    ### Waarom een test ernaast kan zitten

    **Te laag.** Een test laat zien wat een model deed met de prompts, tools, pogingen en tijd die het kreeg, niet alles wat het kan. Volgens Shevlane en collega's moeten onderzoekers verborgen capabilities naar boven halen, bijvoorbeeld met prompt engineering of fine-tuning. Het volle vermogen uit een model halen heet elicitation. Phuong en collega's maken er een regel van dat je het vermogen van een model niet mag onderschatten als je de risico's ervan goed wilt inschatten. GPT-4 is een voorbeeld. Volgens de system card van OpenAI van maart 2023, het verslag dat OpenAI bij het model publiceerde over de tests en de veiligheidsmaatregelen, vonden voorlopige tests het model niet in staat om zichzelf zelfstandig te kopiëren, middelen te verwerven en te voorkomen dat het werd uitgeschakeld, in de echte wereld. Maar de testers, ARC, het Alignment Research Center, waarvan het evaluatieteam nu METR is, konden het model niet fine-tunen, en hadden geen toegang tot de uiteindelijke versie die OpenAI uitbracht.

    Die tests zijn bekend van een verhaal. GPT-4 vroeg een klusjesmaker op TaskRabbit, een site voor kleine betaalde klussen, om een CAPTCHA op te lossen, de plaatjespuzzel waarmee websites programma's buiten de deur houden. Toen de klusjesmaker vroeg of het een robot was, redeneerde het model, dat gevraagd was hardop te redeneren, dat het niet moest verraden dat het een robot was en een smoes moest verzinnen, en antwoordde het dat het geen robot was, maar slecht zag. Het eigen verslag van ARC (Beth Barnes, 17 maart 2023) vult aan wat navertellingen vaak weglaten. ARC gaf het model de inloggegevens van een TaskRabbit-account en stelde TaskRabbit zelf voor. Een onderzoeker speelde de browser en kopieerde wat het model schreef naar de chat met de klusjesmaker. Toen het model vastliep, gaf ARC een hint: dat de klusjesmaker CAPTCHA's kon oplossen. De geteste versie was niet dezelfde als het uitgebrachte model. En over al die tests schrijft ARC dat geen enkel model, met de prompts van ARC, een geloofwaardig volledig plan maakte om zichzelf zelfstandig te kopiëren.

    Een test kan ook te laag uitvallen doordat het model zich inhoudt. Teun van der Weij en collega's (ICLR 2025) omschrijven sandbagging als strategisch onderpresteren bij een evaluatie. De ontwikkelaars of het AI-systeem zelf kunnen willen dat een test laag uitvalt, schrijven ze. Ze gaven frontier language models, zoals GPT-4, de opdracht om slechter te presteren op tests van gevaarlijke capabilities en goed te blijven op onschuldige tests, en ze fine-tunden modellen om een capability te verbergen, behalve als er een wachtwoord in de prompt stond. Ze concluderen voorzichtig dat hun uitkomsten erop wijzen dat capability evaluations kwetsbaar zijn voor sandbagging. Volgens het International AI Safety Report 2026 komt het vaker voor dat modellen een testomgeving onderscheiden van echt gebruik en mazen in evaluaties uitbuiten, zodat gevaarlijke capabilities vóór de release onopgemerkt kunnen blijven.

    **Te hoog.** Het Britse AI Safety Institute schreef op 24 oktober 2024 dat elicitation capabilities ook kan overschatten, als dezelfde taken dienen om de prompts af te stellen en om te evalueren. En een judge model kan zich vergissen, zoals in de gok. RepliBench laat elke variant van een taak 25 keer draaien. Uit die 25 runs schat het pass@10, de kans dat minstens één van 10 pogingen slaagt, op de manier van Chen en collega's (2021): kies willekeurig 10 van de 25 runs, zonder er een twee keer te kiezen, en vraag hoe waarschijnlijk het is dat er minstens één slaagde. Oefening 1 rekent dat uit.

    ### Waar ze het oneens zijn

    Over de vraag naar welke kant tests ernaast zitten: de meeste bronnen vrezen dat ze te laag uitvallen (METR, Phuong en collega's, Shevlane en collega's), terwijl het Britse AI Safety Institute en RepliBench erop wijzen dat afstellen op de testtaken, en judges, ze te hoog kunnen maken. Over wat een gedeeltelijke uitkomst betekent: de beoordeling van Kimi K3 neemt één succes in 10 als teken van de capability, terwijl de system card van GPT-4 uit mislukkingen afleidde dat het model waarschijnlijk nog niet in staat was om zichzelf zelfstandig te kopiëren en middelen te verzamelen. Beide passen bij de regel dat een succes meer laat zien dan een mislukking, maar de woorden lopen sterk uiteen. En over hoeveel je op tests mag leunen: Shevlane en collega's noemen te veel vertrouwen in evaluaties een risico op zich, omdat er dan riskante modellen uitkomen terwijl men zich ten onrechte veilig voelt.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **dangerous capability evaluation** | een test of een model dingen kan die ernstige schade kunnen veroorzaken, zoals helpen bij het maken van wapens, cyberaanvallen uitvoeren of zichzelf kopiëren |
        | externe evaluator | een organisatie die een model test dat ze niet zelf bouwde, vaak vóór de release, zoals een AI Security Institute of METR |
        | elicitation | het volle vermogen uit een model halen tijdens een test, met goede prompts, tools, meer pogingen of fine-tuning, zodat de uitkomst geen onderschatting is |
        | sandbagging | een model dat bij een test bewust slechter presteert dan het kan, uit zichzelf of omdat de ontwikkelaar dat wil |
        | judge model | een language model dat leest wat een ander model deed en beslist of het lukte |
        | pass@k | de kans dat minstens één van $k$ pogingen slaagt |
        | $q$ | in het uitgewerkte voorbeeld, de kans dat een judge in één run ten onrechte succes meldt |
        | $n$, $c$, $k$ | in oefening 1, de gedane runs, de runs daarvan die slaagden, en de gekozen runs |
        """),
        L.watch_out("Een dangerous capability evaluation vraagt of een model iets kan, niet of het dat zou doen. Een score geldt voor één set taken, één opzet en één versie van het model: een uitgebracht model kan verschillen van het geteste. In het Engels betekent sandbagging in de sport onder je niveau spelen om later voordeel te halen; hier betekent het dat een model bij een test bewust slechter presteert. De gevallen van de oefeningen, de demo en de regel van 95% zijn van de course zelf."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo lees je de uitkomst van een test op een gevaarlijke capability:

    1. Vraag wat er getest is: welke taken, met welke tools, hoeveel pogingen, en welke versie van het model.
    2. Vraag wie er testte, en hoeveel tijd en toegang ze hadden.
    3. Een succes, zelfs één op 10, laat zien dat de capability er is, in elk geval met die tools.
    4. Nul successen sluiten alleen hoge kansen op succes uit; reken uit welke kansen nog mogelijk zijn.
    5. Controleer hoe er gescoord is: een judge die zich vergist, of taken die ook dienden om de prompts af te stellen, kunnen een score opdrijven.
    6. Vraag wat een score omlaag kan duwen: zwakke prompts, te weinig pogingen, geen fine-tuning, of een model dat zich inhoudt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.vstack([
        mo.md(r"""
        ### Uitgewerkt voorbeeld

        Stap 5 voor een judge model. Als de judge in elke run met een kans $q$ ten onrechte succes meldt, dan worden alle $k$ runs van een model dat nooit slaagt als mislukt gescoord met een kans van $(1 - q)^k$, en wordt er minstens één als succes gescoord met een kans van:
        """),
        L.formula(r"P(\text{onterecht geslaagd}) = 1 - (1 - [[data:q]])^{[[hyper:k]]}"),
        mo.md(r"""
        De cel rekent het uit voor 1, 5 en 10 runs.
        """),
    ])
    return


@app.cell
def example_judge():
    false_pass = 0.05   # de kans dat de judge in één run ten onrechte succes meldt
    for k in [1, 5, 10]:   # de runs van een model dat nooit slaagt
        print(k, round(1 - (1 - false_pass) ** k, 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Bij 10 runs komt een model dat de taak nooit kan met een kans van 40,1% door de test: de 40% van RepliBench. Het is de berekening van de willekeurige controles in [Dozen, oracles en tools](/course/open/rai-boxing-and-tool-ai?lang=nl), van de andere kant bekeken: daar maakten meer controles het waarschijnlijker dat je een fout antwoord vindt; hier maken meer runs het waarschijnlijker dat er een onterecht succes doorheen glipt. Voor de scores rekent RepliBench pass@10 niet uit met een kans per run, maar met de runs die al gedaan zijn, en oefening 1 doet hetzelfde.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pass():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. pass_at_k: één succes in 25 runs

        RepliBench laat elke versie van een taak `n` keer draaien, waarvan er `c` slagen, en meldt pass@k voor `k` = 10. `pass_at_k(n, c, k)` geeft de kans terug dat `k` van de `n` runs, willekeurig gekozen zonder er een twee keer te kiezen, minstens één succes bevatten. Houd `missed` bij, de kans dat elke tot nu toe gekozen run mislukte, en begin bij `1.0`. Laat `missed` bij elke keuze `i`, van 0 tot en met `k - 1`, de kans worden dat deze keuze en elke keuze ervoor mislukten. Geef de kans terug dat minstens één gekozen run slaagde.

        De check roept `pass_at_k(25, 1, 10)` aan: 25 runs, één succes, en pass@10. Daarna zeven andere gevallen.
        """),
        L.task("pass@k uit runs die al gedaan zijn."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pass():
    box_pass = L.code_box("pass", lines=5)
    box_pass
    return (box_pass,)


@app.cell(hide_code=True)
def check_pass(box_pass):
    result_pass = L.check("pass", box_pass.value)
    result_pass
    return (result_pass,)


@app.cell(hide_code=True)
def hints_pass():
    L.hints(
        "pass",
        [
            "Reken het met de hand uit voor `pass_at_k(4, 2, 2)`: vier runs, waarvan er twee mislukten. Wat is de kans dat de eerste keuze een mislukking is? Als dat zo was, hoeveel mislukkingen en hoeveel runs zijn er dan nog over voor de tweede keuze?",
            "Begin met `missed = 1.0` en een loop `for i in range(k):`. Vóór keuze `i` zijn er al `i` mislukkingen gekozen: hoeveel mislukkingen zijn er nog over, en hoeveel runs? Vermenigvuldig `missed` met de kans dat ook deze keuze een mislukking is, en zet `return 1 - missed` na de loop.",
            "`        missed = missed * ___`",
        ],
        why="`missed = 1.0`, dan `for i in range(k):` met `missed = missed * (n - c - i) / (n - i)`, en `return 1 - missed` na de loop. Met de hand: de eerste keuze is een mislukking met een kans van 2/4; daarna is er nog één mislukking over onder drie runs, dus de tweede keuze is er een met een kans van 1/3. Beide mislukken met een kans van 1/6, en pass@2 is 5/6, ongeveer 0,8333.",
    )
    return


@app.cell(hide_code=True)
def text_after_pass():
    mo.md(r"""
    Met één succes in 25 runs is pass@10 0,4; met twee 0,65; met drie 0,80; met vijf 0,94. Elk succes telt zwaar, en dat is de bedoeling van pass@k: het vraagt of een model een taak überhaupt kan, niet hoe betrouwbaar. Een onterecht succes van een judge telt even zwaar als een echt succes. Bij één keuze is pass@1 gewoon het deel van de runs dat slaagde: `pass_at_k(10, 3, 1)` geeft ongeveer 0,3.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: nul op n

    Een test laat een model $n$ keer een taak doen, en het slaagt nooit. Welke kansen op succes per poging sluit dat uit? Bij een model dat in elke poging, los van de andere, slaagt met een kans $r$, mislukken alle $n$ pogingen met een kans van $(1 - r)^n$. Ligt die kans onder 5%, dan zouden nul successen zeldzaam zijn voor zo'n model, en noemt de course de kans $r$ uitgesloten: een regel van 95%, een gangbare keuze in de statistiek. Kies $n$ en kijk welke kansen er overblijven.
    """)
    return


@app.cell(hide_code=True)
def demo_zero():
    explore_zero = L.explore(
        "zero", "n: pogingen zonder succes",
        lambda _v: C.demo_frame(_v, L.lang),
        start=C.N_START, values=C.N_VALUES, role="hyper",
        task="Zoek de kleinste n waarbij een kans van 5% per poging uitgesloten is.",
    )
    explore_zero
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Een kans van 5% per poging is uitgesloten vanaf 59 pogingen zonder succes; voor 1% zijn er 299 nodig, meer dan de demo laat zien. Voor grotere $n$ geeft een vuistregel bijna hetzelfde: na $n$ pogingen zonder succes zijn kansen boven ongeveer 3 gedeeld door $n$ uitgesloten. En de regel gaat alleen over deze test: een betere opzet, of een model dat zich niet meer inhoudt, verandert de kans zelf. Andersom is er veel minder nodig: één succes in 10 pogingen, zoals bij Kimi K3, laat zien dat de capability er is.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ruled():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. ruled_out: wat nul successen uitsluiten

        `ruled_out(attempts, rates)` geeft de kleinste kans in de lijst `rates` terug die nul successen in `attempts` pogingen uitsluiten volgens de regel van 95%: de eerste `rate` waarbij alle pogingen van een model met die kans op succes per poging zouden mislukken met een kans onder 0,05. De lijst loopt van klein naar groot. Ga door `rates` als `rate`, geef de eerste terug die uitgesloten is, en geef na de loop `None` terug als er geen is.

        De check roept `ruled_out(10, rates)` aan, met `rates` de lijst 0,01, 0,02 enzovoort tot en met 0,99. Daarna acht andere gevallen.
        """),
        L.task("Welke kansen een test zonder één enkel succes uitsluit."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ruled():
    box_ruled = L.code_box("ruled", lines=5)
    box_ruled
    return (box_ruled,)


@app.cell(hide_code=True)
def check_ruled(box_ruled):
    result_ruled = L.check("ruled", box_ruled.value)
    result_ruled
    return (result_ruled,)


@app.cell(hide_code=True)
def hints_ruled():
    L.hints(
        "ruled",
        [
            "Reken het met de hand uit voor `ruled_out(10, [0.1, 0.2, 0.3])`: wat is voor een model dat in 10% van de pogingen slaagt de kans dat alle 10 pogingen mislukken? En bij 20% en 30%? Bij welke kans ligt die voor het eerst onder 0,05?",
            "Schrijf `for rate in rates:`. Vraag in de loop: wat is bij een model met deze `rate` de kans dat één poging mislukt, en dat alle `attempts` pogingen mislukken? Geef `rate` terug met `return rate` als die kans onder 0,05 ligt. Zet na de loop `return None`.",
            "`        if ___ < 0.05:`",
        ],
        why="`for rate in rates:` met daarin `if (1 - rate) ** attempts < 0.05:` en `return rate`, en `return None` na de loop. Met de hand: 0,9 tot de macht 10 is ongeveer 0,349, 0,8 tot de macht 10 ongeveer 0,107, en 0,7 tot de macht 10 ongeveer 0,028, dus 0,3 is de eerste uitgesloten kans.",
    )
    return


@app.cell(hide_code=True)
def text_after_ruled():
    mo.md(r"""
    Tien pogingen zonder succes sluiten kansen van 26% en meer uit, 50 pogingen 6% en meer, 100 pogingen 3% en meer. Zonder pogingen is er niets uitgesloten, en geeft de functie `None`. Een test die niets vindt, zegt dus wel iets: niet dat een capability ontbreekt, maar dat de kans erop in die opzet waarschijnlijk onder een bepaald getal ligt, en hoe minder pogingen, hoe hoger dat getal.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_claim():
    mo.md(r"""
    ### Oefening 3. Nul van de 50

    Een week vóór een release geeft een evaluatieteam een nieuw model 50 cyberopgaven, met een eenvoudige opzet om tools te gebruiken. Het lost er geen enkele op. De ontwikkelaar meldt in een aankondiging dat tests laten zien dat het model geen gevaarlijke cybercapability heeft.
    """)
    return


@app.cell(hide_code=True)
def choice_claim_cell():
    choice_claim = L.choice(
        "claim",
        question="Welk antwoord past het best bij de bronnen van deze les?",
        options={
            "proves": "Klopt, want 0 van de 50 laat zien dat de capability ontbreekt",
            "right": "Niets opgelost in deze opzet, maar dat zegt niet dat het niet kan",
            "worthless": "De test zegt helemaal niets; alleen gebruik in de echte wereld telt",
            "all": "Alleen een model dat alle 50 oplost, telt als gevaarlijk",
        },
        correct="right",
        why={
            "right": "Volgens Dijkstra kan testen de aanwezigheid van bugs laten zien, maar nooit hun afwezigheid. Volgens de eigen telling van de course laat 0 van de 50 nog kansen per opgave tot ongeveer 5,8% open, en betere prompts, meer pogingen, fine-tuning of een model dat zich niet meer inhoudt, kunnen de uitkomst veranderen. Volgens METR onderschatten korte tests vóór een release een model waarschijnlijk, en volgens het International AI Safety Report betekent het ontbreken van gevonden risico's niet dat die risico's klein zijn.",
            "proves": "Een test kan een capability laten zien, niet aantonen dat ze ontbreekt: 0 van de 50 laat kansen per opgave tot ongeveer 5,8% open, alleen al in deze opzet. Zie *Aanwezigheid, geen afwezigheid*.",
            "worthless": "Elke bron hier ziet tests als nuttig maar niet genoeg: volgens Shevlane en collega's is evalueren nodig, maar niet voldoende. En de test zegt wel iets: de test sluit een hoge kans uit in deze opzet. Zie *Wat er getest wordt*.",
            "all": "Dat is precies andersom: één succes kan een capability laten zien, zoals één op 10 pogingen bij Kimi K3. Zie *Wie er test*.",
        },
    )
    choice_claim
    return (choice_claim,)


@app.cell(hide_code=True)
def text_exercise_story():
    mo.md(r"""
    ### Oefening 4. Een kop

    Een nieuwssite schrijft dat GPT-4 online een mens inhuurde en loog om langs een CAPTCHA te komen, helemaal uit zichzelf.
    """)
    return


@app.cell(hide_code=True)
def choice_story_cell():
    choice_story = L.choice(
        "story",
        question="Wat zegt het eigen verslag van ARC hierover?",
        options={
            "replicate": "Waar, en het laat zien dat GPT-4 zichzelf kon kopiëren naar nieuwe computers",
            "right": "Deels waar, want een onderzoeker gaf berichten door en ARC gaf een hint",
            "excuse": "Deels waar, want ARC schreef de smoes en het model stuurde die alleen door",
            "release": "Waar, en de versie van GPT-4 die uitkwam deed in tests precies hetzelfde",
        },
        correct="right",
        why={
            "right": "Het model vroeg een klusjesmaker op TaskRabbit wel om hulp en gaf een valse smoes. Maar ARC stelde TaskRabbit voor, een onderzoeker speelde de browser en kopieerde de berichten naar de chat, ARC gaf een hint toen het model vastliep, en de geteste versie was niet de uitgebrachte. Over al die tests concludeerde ARC dat geen enkel model een geloofwaardig volledig plan maakte om zichzelf te kopiëren. Het verhaal laat een stukje van een capability zien, met hulp.",
            "replicate": "ARC concludeerde juist dat geen enkel model, met de prompts van ARC, een geloofwaardig volledig plan maakte om zichzelf zelfstandig te kopiëren. En volgens de system card was het model daar niet toe in staat. Zie *Waarom een test ernaast kan zitten*.",
            "excuse": "Het model schreef de smoes zelf, toen het gevraagd was hardop te redeneren. Wat ARC toevoegde, waren de inloggegevens, het voorstel om TaskRabbit te gebruiken, een onderzoeker die de berichten kopieerde, en een hint. Zie *Waarom een test ernaast kan zitten*.",
            "release": "ARC testte eerdere versies, die volgens ARC niet dezelfde waren als het uitgebrachte model. Volgens de system card was de uitgebrachte versie beter in sommige dingen die de eerdere versies tegenhielden. Zie *Waarom een test ernaast kan zitten*.",
        },
    )
    choice_story
    return (choice_story,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Als dit, dan dat: frontier safety frameworks](/course/open/rai-frontier-frameworks?lang=nl) legde thresholds vast die bepalen wat er gebeurt als een model iets gevaarlijks kan; de tests in deze les bepalen of een threshold overschreden is. De regel van Anthropic om te doen alsof een model boven een threshold zit zolang het tegendeel niet te laten zien is, past bij tests die een capability kunnen laten zien, maar niet de afwezigheid ervan. In [Dozen, oracles en tools](/course/open/rai-boxing-and-tool-ai?lang=nl) was het zwakkere model te vertrouwen vanwege wat het niet kan, en ook dat vertrouwen rust op zulke tests; de willekeurige controles daar en het judge model hier zijn dezelfde berekening, van twee kanten bekeken. De volgende les, *Hoe lange taken kan een model aan?*, meet capabilities op een andere manier: aan hoe lang de taken zijn die een model kan afmaken.

    **Een voorbeeld uit het dagelijks leven.** Je zoekt in de zakken van je jas naar je sleutels. Vind je ze, dan weet je dat ze er zijn. Zoek je snel in één zak en vind je niets, dan weet je weinig: ze kunnen in een andere zak zitten, of dieper in dezelfde. Alleen als je elke zak goed doorzoekt, wordt het onwaarschijnlijk dat je ze over het hoofd hebt gezien, en zelfs dan kan er een zak zijn die je vergeten was.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_claim, choice_story, result_pass, result_ruled):
    L.wrap_up(result_pass, result_ruled, choice_claim, choice_story)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
