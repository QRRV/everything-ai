# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-frontier-frameworks. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-frontier-frameworks", "nl")
    C = L.checks   # de capability thresholds van een verzonnen lab, de scores van een nieuw model, en tien versies van een model


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Open weights: voor en tegen](/course/open/rai-open-weights?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_marginal():
    quick_marginal = L.choice(
        "quick-marginal",
        question="Een release verhoogt de kans op een dreiging in een jaar van 0,2 naar 0,25, en één zo'n gebeurtenis kost 40 schadepunten. Wat is het marginal risk van de release voor deze dreiging?",
        options={
            "right": "2 schadepunten",
            "with": "10 schadepunten: het risico met de release",
            "chance": "0,05, de stijging van de kans",
        },
        correct="right",
        why={
            "right": "Het risico met de release is 0,25 × 40 = 10, het risico zonder 0,2 × 40 = 8. De release voegt het verschil toe: 2 schadepunten.",
            "with": "10 is het hele risico met de release, maar 8 daarvan was er zonder de release ook al. Het marginal risk is wat de release toevoegt: 10 − 8 = 2. Zie [Open weights: voor en tegen](/course/open/rai-open-weights?lang=nl).",
            "chance": "Risico is kans maal schade, dus de stijging van de kans moet je nog met de schade vermenigvuldigen: 0,05 × 40 = 2 schadepunten. Zie [Open weights: voor en tegen](/course/open/rai-open-weights?lang=nl).",
        },
    )
    quick_marginal
    return (quick_marginal,)


@app.cell(hide_code=True)
def quick_check_reference():
    quick_reference = L.choice(
        "quick-reference",
        question="Kapoor en anderen vergelijken een open model met technologie die er al was, met closed models of met andere vergelijkingspunten. Waarom doet die keuze ertoe?",
        options={
            "right": "Dezelfde release voegt veel of weinig toe, afhankelijk van wat er al is",
            "none": "Het doet er niet toe: het risico van een model blijft hetzelfde, waarmee je het ook toevallig vergelijkt",
            "closed": "Omdat closed models het enige vergelijkingspunt zijn dat je kunt meten",
        },
        correct="right",
        why={
            "right": "Het marginal risk is het risico met de release min het risico zonder, en de wereld zonder de release is het vergelijkingspunt. Tegenover een wereld zonder AI kan een release veel toevoegen; tegenover een wereld die dezelfde hulp al heeft, weinig.",
            "none": "Het totale risico van een wereld met het model blijft misschien gelijk, maar wat de release toevoegt, hangt af van de wereld zonder: met of zonder andere modellen, met of zonder een zoekopdracht op het web. Zie [Open weights: voor en tegen](/course/open/rai-open-weights?lang=nl).",
            "closed": "Closed models zijn maar één vergelijkingspunt: Kapoor en anderen noemen ook technologie die er al was, en andere. De keuze doet ertoe omdat dezelfde release veel of weinig toevoegt, afhankelijk van wat er al is. Zie [Open weights: voor en tegen](/course/open/rai-open-weights?lang=nl).",
        },
    )
    quick_reference
    return (quick_reference,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In de vorige les ging de vraag over het vrijgeven van één model. AI-bedrijven krijgen zo'n vraag bij elk nieuw model, en elke keer kan het model meer. Op dat moment beslissen is moeilijk: het model is klaar, concurrenten zitten er vlak achter, en het gevaar is nog onzeker. Daarom beslissen sommige bedrijven vooraf: als een model dit kan, dan doen we dat. Deze les laat zien wat zulke beloftes zeggen, hoe je ze voor een nieuw model controleert, en waar ze tekortschieten.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Op de AI Seoul Summit in mei 2024 beloofden zestien AI-bedrijven een safety framework voor ernstige risico's te publiceren, en later sloten er nog vier aan. Hoeveel bedrijven hadden er in december 2025 een gepubliceerd, volgens de telling van METR, een organisatie die AI-modellen evalueert?",
        options={
            "all": "Alle 20, want ze hadden het beloofd",
            "right": "12",
            "few": "3: de grootste labs",
        },
        reveal="**12.** In december 2025 schreef METR dat er op dat moment twaalf bedrijven een frontier safety framework hadden gepubliceerd. Capability thresholds stonden in 9 daarvan, en 8 noemden voorwaarden waaronder plannen voor verdere ontwikkeling stilgelegd worden. METR voegt eraan toe dat sommige andere bedrijven, zoals IBM en Samsung, vergelijkbare stukken publiceerden. Een belofte om te publiceren leidde niet altijd tot een framework, en een framework heeft niet altijd thresholds.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Als dit, dan dat

    Anthropic beschrijft zijn Responsible Scaling Policy als een vrijwillig framework om catastrofale risico's van geavanceerde AI-systemen te beheersen. OpenAI noemt zijn versie het Preparedness Framework, Google DeepMind de zijne het Frontier Safety Framework. De algemene naam is een **frontier safety framework**: een beleid waarin een bedrijf dat de krachtigste modellen bouwt vooraf zegt op welke gevaarlijke vaardigheden het test, en wat het doet als een model ze heeft.

    Holden Karnofsky (2024) beschrijft de kern: het zijn beloftes van de vorm: als een AI-model vaardigheid X heeft, dan moeten maatregelen Y klaarstaan, en zo nodig stelt het bedrijf het uitbrengen of de ontwikkeling uit tot die maatregelen er op tijd zijn. Karnofsky noemt ze **if-then commitments**. Daarvoor hoeft volgens Karnofsky niemand het eens te zijn over de vraag of grote AI-risico's al dichtbij zijn: over de "dan" kun je het vooraf eens worden.

    Op de AI Seoul Summit van 21 mei 2024 beloofden 16 bedrijven zo'n framework te publiceren, gericht op ernstige risico's, vóór de volgende AI-top in Frankrijk. Ze zouden thresholds vastleggen waarbij ernstige risico's van een model of systeem als ontoelaatbaar gelden, tenzij die risico's genoeg beperkt worden. En in het uiterste geval zouden ze een model of systeem helemaal niet ontwikkelen of uitbrengen, als maatregelen de risico's niet onder die thresholds kunnen houden.

    ### De "als": een capability threshold

    De "als" is een **capability threshold**. Het framework van OpenAI, versie 2 van april 2025, omschrijft capability thresholds als concrete beschrijvingen van wat een AI-systeem iemand zou kunnen helpen doen, of zelf zou kunnen doen, waardoor het risico op ernstige schade flink stijgt. Met ernstige schade bedoelt OpenAI de dood of zware verwonding van duizenden mensen, of honderden miljarden dollars aan economische schade. Er zijn twee niveaus: High, voor vaardigheden die bestaande wegen naar ernstige schade flink versterken, en Critical, voor een reëel risico op een nieuw soort dreiging waarvoor nog geen voorbeeld bestaat. Versie 3.1 van Google DeepMind, van april 2026, noemt zijn thresholds Critical Capability Levels, of CCL's: niveaus waarop modellen zonder maatregelen een verhoogd risico op ernstige schade kunnen geven.

    Een threshold uit versie 3.4 van Anthropic, van juli 2026, gaat over de productie van chemische of biologische wapens die al bekend zijn: AI-systemen die mensen of groepen met een technische basisopleiding, zoals een bachelor in een bètavak, flink helpen om zulke wapens te maken of te krijgen en in te zetten, met kans op catastrofale schade. Andere thresholds gaan over cyberaanvallen, of over AI die zelf AI-onderzoek doet, zoals het eerste niveau voor ML R&D van Google DeepMind: het werk van elk team van onderzoekers bij Google dat AI beter maakt volledig kunnen automatiseren, voor ongeveer dezelfde kosten.

    Of een model een threshold haalt, ziet een lab met een evaluatie: een test van het model op taken uit dat domein. Een latere les, *Testen op gevaarlijke vaardigheden*, laat zien hoe. In deze les geeft zo'n test een score van 0 tot en met 1.

    ### De "dan": wat er moet gebeuren

    OpenAI brengt geen modellen uit die een High capability threshold halen, tot de risico's ervan voldoende beperkt zijn. Bij Critical zegt de tabel met safeguards: zolang er geen safeguards en beveiliging zijn vastgelegd die aan de Critical-standaard voldoen, stopt de verdere ontwikkeling. Bij Google DeepMind gaat een model pas naar buiten als het bevoegde orgaan binnen het bedrijf het overgebleven risico aanvaardbaar vindt, met een safety case als een CCL bereikt is. Een safety case is een geschreven redenering, met onderbouwing, dat het risico aanvaardbaar is.

    En als de tests niet kunnen laten zien dat een model ver genoeg onder een threshold blijft? Versie 2.2 van Anthropic, van mei 2025, zei dat Anthropic dan zou doen alsof het model de Capability Threshold voorbij was. Wie niet kan laten zien dat een model ver genoeg onder een threshold blijft, behandelt het als erboven.

    ### Waarschuwen vóór de threshold

    Google DeepMind legt lijnen onder zijn thresholds: Alert Thresholds, net iets eerder dan de CCL's. Een model dat zo'n alertlijn passeert, heeft de threshold nog niet bereikt, maar komt in de buurt, en de safeguards die bij de threshold horen, kunnen op tijd klaargemaakt worden.

    ### Waar het moeilijk wordt

    De bedrijven schrijven de frameworks zelf, en Anthropic noemt het zijne vrijwillig. Karnofsky, die ervoor pleit, schrijft dat vrijwillige beloftes op zichzelf het risico waarschijnlijk niet laag houden, en dat beloftes van de vorm als-dan niet bij alle risico's van AI passen. Anthropic schreef in 2026, na ermee gewerkt te hebben, dat vooraf vastgelegde niveaus veel vager bleken dan verwacht. Soms kwamen de modellen duidelijk in de buurt van een threshold, maar bleef onzeker of ze die echt voorbij waren: het evalueren van modellen is volgens Anthropic nog niet ver genoeg om daar een sluitend antwoord op te geven. In zulke gevallen zette Anthropic de safeguards toch in.

    Anthropic en Google DeepMind schrijven allebei dat één bedrijf alleen ze niet kan laten werken. Versie 3 van Anthropic koos een andere koers, vanwege een collectief actieprobleem: als één ontwikkelaar pauzeert om veiligheidsmaatregelen in te bouwen terwijl anderen doorgaan zonder sterke maatregelen, kan de wereld er minder veilig van worden, want dan bepalen de ontwikkelaars met de zwakste bescherming het tempo. Anthropic noemt nu aanbevelingen voor de hele sector, maar schrijft dat het niet kan beloven die als enige te volgen. Google DeepMind schrijft dat sommige van zijn maatregelen de samenleving alleen helpen als alle relevante organisaties vergelijkbare bescherming bieden. En het redeneert met marginal risk, zoals in de vorige les: als andere modellen even sterk zijn en weinig maatregelen hebben, voegt het eigen uitbrengen waarschijnlijk weinig risico toe.

    De frameworks veranderen ook vaak. Dat van Anthropic had 9 versies tussen september 2023 en juli 2026; dat van Google DeepMind vier sinds mei 2024, en het wordt minstens één keer per jaar herzien. Of ze het bedrijf echt binden, is omstreden. OpenAI schrijft dat het deze zeer krachtige modellen pas uitbrengt als het safeguards heeft gebouwd die de risico's op ernstige schade voldoende beperken. Sam Coggins en anderen antwoorden, in een preprint uit 2025 die nog niet door vakgenoten beoordeeld is, dat het Preparedness Framework om evaluatie van een kleine minderheid van de AI-risico's vraagt, en evaluatie van geen enkel risico eist.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **frontier safety framework** | het beleid van een bedrijf over op welke gevaarlijke vaardigheden van zijn krachtigste modellen het test, en wat het dan doet |
        | **if-then commitment** | een belofte vooraf: als een model deze vaardigheid bereikt, dan moeten deze safeguards klaarstaan, of wacht de ontwikkeling |
        | **capability threshold** | het vooraf vastgelegde niveau van een gevaarlijke vaardigheid waarop de "dan" geldt |
        | alertlijn | een lijn een eindje onder een capability threshold, als vroege waarschuwing; Google DeepMind noemt die een alert threshold |
        | safety case | een geschreven redenering, met onderbouwing, dat het risico van een model aanvaardbaar is |
        | evaluatie | een test van wat een model kan in een domein; hier loopt de score van 0 tot en met 1 |
        """),
        L.watch_out("Een framework is een belofte van het bedrijf zelf: het bedrijf schrijft het, kan het veranderen, en beslist zelf of een threshold bereikt is. En een threshold is maar zo goed als de tests waarmee je meet of die bereikt is."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Kies de domeinen waarin een model kan helpen ernstige schade aan te richten, zoals biologie, cyberaanvallen of AI-onderzoek.
    2. Leg per domein een capability threshold vast: wat een model dan zou kunnen, en welke score op je tests dat laat zien.
    3. Leg de "dan" vast: welke safeguards er moeten zijn voordat je het model uitbrengt, en wanneer de ontwikkeling stopt.
    4. Zet een alertlijn een eindje onder elke threshold, zodat je op tijd ziet dat een model in de buurt komt.
    5. Test elke nieuwe versie. Een domein dat je niet kon testen, telt als bereikt.
    6. Publiceer het framework, zeg wanneer en waarom je het verandert, en kijk of anderen vergelijkbare beloftes nakomen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een verzonnen lab heeft voor cyberaanvallen een capability threshold van 0,7, op een eigen test. Drie versies van zijn model zijn getest. De cel vergelijkt elke versie met de threshold. Alle getallen zijn verzonnen.
    """)
    return


@app.cell
def example_threshold():
    threshold = 0.7                       # de capability threshold van het lab voor cyberaanvallen, verzonnen
    scores = [0.48, 0.63, 0.74]           # drie versies van het model op de cybertest, verzonnen
    for version in range(len(scores)):
        if scores[version] >= threshold:
            print("versie", version + 1, "heeft de threshold bereikt: eerst safeguards")
        else:
            print("versie", version + 1, "zit onder de threshold")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Versie 1 en 2 zitten onder 0,7, dus de "dan" geldt nog niet, al zit versie 2 er niet ver vanaf. Versie 3 bereikt de threshold met 0,74: voordat die versie uitgebracht wordt, moeten de beloofde safeguards er zijn. In de oefeningen controleer je meer domeinen tegelijk, met een regel voor een domein dat niet getest is, en zet je een alertlijn onder de threshold.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_reached():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. reached: welke thresholds zijn bereikt?

        Het framework van het verzonnen lab heeft per domein een capability threshold, in een dict `thresholds`, zoals `{"bio": 0.6, "cyber": 0.7, "autonomy": 0.5}`. De scores van een nieuw model staan in een dict `scores`; een domein dat niet getest is, ontbreekt daarin. Schrijf `reached(scores, thresholds)`. Begin een lijst `hits`, en loop met een `for`-loop langs `thresholds.items()`, met `domain` en `threshold` voor elk domein. Houd elk `domain` over dat niet in `scores` staat, of waarvan de score op de threshold of erboven ligt, en geef `hits` terug. Vervang de `...`. De check roept `reached({"bio": 0.62, "cyber": 0.55}, {"bio": 0.6, "cyber": 0.7, "autonomy": 0.5})` aan, en daarna andere gevallen.
        """),
        L.task("De domeinen waarvan het nieuwe model de threshold bereikt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_reached():
    box_reached = L.code_box("reached", lines=6)
    box_reached
    return (box_reached,)


@app.cell(hide_code=True)
def check_reached(box_reached):
    result_reached = L.check("reached", box_reached.value)
    result_reached
    return (result_reached,)


@app.cell(hide_code=True)
def hints_reached():
    L.hints(
        "reached",
        [
            "Loop langs de domeinen van het framework, niet langs die van de scores: een domein dat niet getest is, heeft geen score, en telt toch mee.",
            "Begin `hits` als een lege lijst. Vraag in een `for`-loop over `thresholds.items()` of `domain` niet in `scores` staat, of anders of de score `threshold` haalt; voeg in beide gevallen `domain` toe. Geef na de loop `hits` terug.",
            "`        if domain not in scores or ___:`",
        ],
        why="`hits = []`, dan `for domain, threshold in thresholds.items():` met `if domain not in scores or scores[domain] >= threshold:` en `hits.append(domain)`; na de loop `return hits`. Omdat `or` stopt zodra het eerste deel waar is, zoekt Python `scores[domain]` nooit op voor een domein dat ontbreekt. Voor het nieuwe model van het lab: biologie 0,62 is minstens 0,6, cyber 0,55 ligt onder 0,7, en autonomie is niet getest: `[\"bio\", \"autonomy\"]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_reached():
    mo.md(r"""
    Voor het nieuwe model van het lab is biologie bereikt met 0,62, cyber niet met 0,55, en autonomie telt als bereikt omdat het niet getest is. Die laatste regel volgt versie 2.2 van Anthropic: een lab dat niet kan aantonen dat een model ver genoeg onder een threshold blijft, doet alsof het model erboven zit. Zonder die regel hoeft een lab dat een moeilijke test overslaat nooit iets te doen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_statuses():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. statuses: onder, alert of bereikt

        Zet nu een alertlijn onder de threshold. Schrijf `statuses(scores, threshold, margin)`. `scores` is een lijst met één score per versie van een model, `threshold` de capability threshold, en `margin` hoe ver de alertlijn onder de threshold ligt. Geef een lijst `result` terug met voor elke `score` de tekst "reached" als die op de threshold of erboven ligt, "alert" als die hooguit `margin` onder de threshold ligt, en anders "below". Gebruik een `for`-loop met `score` voor elke versie, en `if`, `elif` en `else`. Vervang de `...`. De check roept `statuses([0.45, 0.58, 0.66, 0.71], 0.7, 0.1)` aan, en daarna andere gevallen.
        """),
        L.task("Voor elke versie: below, alert of reached."),
    ])
    return


@app.cell(hide_code=True)
def exercise_statuses():
    box_statuses = L.code_box("statuses", lines=10)
    box_statuses
    return (box_statuses,)


@app.cell(hide_code=True)
def check_statuses(box_statuses):
    result_statuses = L.check("statuses", box_statuses.value)
    result_statuses
    return (result_statuses,)


@app.cell(hide_code=True)
def hints_statuses():
    L.hints(
        "statuses",
        [
            "Test eerst het hoogste niveau: een score op de threshold ligt ook boven de alertlijn, en moet \"reached\" krijgen.",
            "Begin `result` als een lege lijst. Voeg in een `for`-loop over `scores` \"reached\" toe als `score` minstens `threshold` is; voeg met `elif` \"alert\" toe als die minstens op de alertlijn ligt, `margin` onder de threshold; voeg met `else` \"below\" toe. Geef na de loop `result` terug.",
            "`        elif score >= ___:`",
        ],
        why="`result = []`, dan `for score in scores:` met `if score >= threshold:` en `result.append(\"reached\")`, `elif score >= threshold - margin:` en `result.append(\"alert\")`, `else:` en `result.append(\"below\")`; na de loop `return result`. In het eerste geval ligt de alertlijn op 0,6: 0,45 en 0,58 liggen eronder, 0,66 ligt tussen de alertlijn en de threshold, en 0,71 heeft 0,7 bereikt.",
    )
    return


@app.cell(hide_code=True)
def text_after_statuses():
    mo.md(r"""
    In het eerste geval slaat versie 3 alarm met 0,66, één versie voordat versie 4 de threshold bereikt met 0,71. Die ene versie is tijd: de safeguards die het framework bij de threshold belooft, kunnen gebouwd worden terwijl het model er nog onder zit.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: waar leg je de alertlijn?

    Tien versies van een verzonnen model worden stap voor stap beter op één test, en de capability threshold ligt op 0,7. Schuif de alertlijn omlaag, en kijk hoeveel versies waarschuwing het lab krijgt voordat de threshold bereikt is.
    """)
    return


@app.cell(hide_code=True)
def demo_margin():
    margin = L.explore("alert-line", "marge onder de threshold", lambda _v: C.demo_frame(_v, L.lang),
                       start=0, values=C.MARGINS, role="",
                       task="Zoek de kleinste marge die minstens twee versies waarschuwing geeft voordat de threshold bereikt is.")
    margin
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Bij een marge van 0,1 slaan versie 7 en 8 alarm voordat versie 9 de threshold bereikt: twee versies om de safeguards klaar te maken; een kleinere marge geeft één versie of geen. Zonder marge is het eerste signaal de threshold zelf, en moeten de safeguards meteen klaar zijn.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_lagging():
    mo.md(r"""
    ### Oefening 3. Een teken van AI die AI verbetert

    Het framework van OpenAI van april 2025 geeft één manier om te zien dat een model de Critical threshold voor AI die AI verbetert heeft bereikt: het model zorgt voor een sprong van een hele modelgeneratie, zoals van OpenAI o1 naar OpenAI o3, in een vijfde van de tijd die dezelfde vooruitgang in 2024 kostte, en dat maandenlang volhouden.
    """)
    return


@app.cell(hide_code=True)
def choice_lagging_cell():
    choice_lagging = L.choice(
        "lagging",
        question="Wat is het nadeel van een threshold die zo gemeten wordt?",
        options={
            "right": "Je weet het pas achteraf, als de snellere vooruitgang al gebeurd is",
            "strict": "Die is zo streng dat geen enkel model er ooit aan komt",
            "safe": "Geen: een threshold die achteraf gemeten wordt, is de veiligste soort, want die rust op feiten",
        },
        correct="right",
        why={
            "right": "De snellere vooruitgang moet eerst gebeuren voordat je die kunt meten, dus tegen de tijd dat dit teken zichtbaar is, was de threshold al bereikt. OpenAI noemt het een lagging indicator, en geeft er een tweede manier naast, een leading indicator: een agent die als onderzoeker beter is dan elk mens, een vaardigheid waar een test in het model zelf naar kan zoeken, voordat er snellere vooruitgang is. Allebei tellen ze als het bereiken van de threshold.",
            "strict": "Streng of niet, het probleem is wanneer je het ziet: de snellere vooruitgang moet eerst gebeuren, dus tegen de tijd dat je die meet, was de threshold al bereikt. OpenAI noemt het een lagging indicator.",
            "safe": "Feiten achteraf komen te laat voor een if-then commitment: de \"dan\" moet klaarstaan voordat het model zover is. OpenAI noemt dit teken een lagging indicator, en geeft er een tweede, een leading indicator, naast.",
        },
    )
    choice_lagging
    return (choice_lagging,)


@app.cell(hide_code=True)
def text_exercise_margin():
    mo.md(r"""
    ### Oefening 4. Een brede alertband

    Een verzonnen lab wil zo veel mogelijk waarschuwing, en legt elke alertlijn ver onder zijn threshold: op de helft ervan.
    """)
    return


@app.cell(hide_code=True)
def choice_margin_cell():
    choice_margin = L.choice(
        "margin",
        question="Wat kost dat het lab?",
        options={
            "right": "Alerts voor veel modellen die nooit een threshold zullen bereiken",
            "none": "Niets: een eerdere waarschuwing geeft alleen meer tijd om je voor te bereiden",
            "lower": "Minder alerts, want alleen de sterkste modellen komen in de buurt van de helft van een threshold",
        },
        correct="right",
        why={
            "right": "Veel modellen komen tot de helft van een threshold en blijven daar. Elke alert vraagt werk, zoals extra tests en safeguards voorbereiden, en een lab dat steeds alerts ziet, kan ze minder serieus gaan nemen. Google DeepMind legt zijn alert thresholds bijvoorbeeld maar net iets eerder dan zijn thresholds.",
            "none": "Een alert vraagt werk, zoals extra tests en safeguards voorbereiden. Op de helft van een threshold slaan veel modellen alarm die nooit in de buurt van de threshold zelf komen. Zie *Waarschuwen vóór de threshold* in *Het idee*.",
            "lower": "De helft van een threshold ligt er ver onder, dus veel modellen komen daar, niet alleen de sterkste, en de meeste komen nooit in de buurt van de threshold zelf. Elk van die modellen slaat alarm, en dat vraagt werk. Zie *Waarschuwen vóór de threshold* in *Het idee*.",
        },
    )
    choice_margin
    return (choice_margin,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Open weights: voor en tegen](/course/open/rai-open-weights?lang=nl) vroeg wat één release toevoegt aan het risico. Deze les liet zien hoe bedrijven vooraf beslissen wat ze doen als een model meer kan: een capability threshold per domein, de safeguards die er dan moeten zijn, een alertlijn eronder, en een regel voor een test die niet gedaan kon worden. Deze les liet ook de grenzen zien: de bedrijven schrijven de frameworks zelf, veranderen ze vaak, en volgens henzelf werken ze alleen als anderen meedoen. De volgende les, *AI in een complex systeem*, kijkt naar wat er gebeurt als veel delen op elkaar reageren; een latere les, *Testen op gevaarlijke vaardigheden*, laat zien waar de scores vandaan komen.

    **Een voorbeeld uit het dagelijks leven.** Een dam heeft een plan dat lang vóór de regen gemaakt is: bij dit waterpeil worden de dorpen stroomafwaarts gewaarschuwd, bij dat peil gaan de overlaten open. Niemand hoeft midden in de storm te beslissen, en het waarschuwingspeil ligt onder het peil waarop het gevaarlijk wordt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_lagging, choice_margin, result_reached, result_statuses):
    L.wrap_up(result_reached, result_statuses, choice_lagging, choice_margin)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
