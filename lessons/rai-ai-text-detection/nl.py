# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-ai-text-detection. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import chat
    from everything_ai.lesson import lesson

    L = lesson("rai-ai-text-detection", "nl")
    C = L.checks   # de run van de course: 20 vragen, een antwoord van een mens en een van Qwen3, gescoord door Qwen3


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Waar komt dit plaatje vandaan?](/course/open/rai-content-provenance?lang=nl) en [Hoe goed is een language model?](/course/open/lm-nll?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_label():
    quick_label = L.choice(
        "quick-label",
        question="Een plaatje heeft Content Credentials die kloppen, ondertekend door een persbureau dat je vertrouwt. Wat vertellen ze je?",
        options={
            "true": "Dat wat het laat zien echt gebeurde",
            "right": "Dat het persbureau tekende en er sindsdien niets veranderde",
            "no-ai": "Dat er geen AI aan te pas kwam: plaatjes van AI hebben zo'n label niet",
        },
        correct="right",
        why={
            "right": "Een label dat klopt, zegt wie voor het plaatje instaat en dat er sindsdien niets veranderde, niet dat het waar is. Tekst heeft meestal helemaal geen label; deze les vraagt wat er dan overblijft.",
            "true": "Wie een onware bewering ondertekent, ondertekent die net zo goed. Zie *Wat een label je niet vertelt* in [Waar komt dit plaatje vandaan?](/course/open/rai-content-provenance?lang=nl).",
            "no-ai": "Een label zegt wie tekende en wat de ondertekenaar beweert; tools die plaatjes met AI maken, kunnen ook labels toevoegen. Zie *Content Credentials* in [Waar komt dit plaatje vandaan?](/course/open/rai-content-provenance?lang=nl).",
        },
    )
    quick_label
    return (quick_label,)


@app.cell(hide_code=True)
def quick_check_nll():
    quick_nll = L.choice(
        "quick-nll",
        question="Onder hetzelfde model heeft tekst A een gemiddelde NLL van 1,2 en tekst B een van 2,9. Wat zegt dat?",
        options={
            "better": "Het model voorspelde B beter dan A",
            "right": "B verraste het model meer, per voorspelling",
            "longer": "Tekst B is langer, want de NLL telt de verrassing van elk token op",
        },
        correct="right",
        why={
            "right": "De gemiddelde NLL is de gemiddelde verrassing per voorspelling: hoe hoger, hoe minder het model de tekst verwachtte. Deze les vraagt of dat getal je vertelt wie een tekst schreef.",
            "better": "Lager is beter: een lagere gemiddelde NLL betekent dat het model de tekst beter voorspelde, en B heeft de hogere. Zie [Hoe goed is een language model?](/course/open/lm-nll?lang=nl).",
            "longer": "Het gemiddelde deelt de totale verrassing door het aantal voorspellingen, dus een tekst scoort niet hoger omdat hij langer is. Zie [Hoe goed is een language model?](/course/open/lm-nll?lang=nl).",
        },
    )
    quick_nll
    return (quick_nll,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In augustus 2023 zette Vanderbilt University de AI-detector van Turnitin uit, een dienst die het werk controleert dat studenten inleveren. Volgens de universiteit claimde Turnitin een false positive rate van 1%. Vanderbilt stuurde in 2022 75.000 werkstukken naar Turnitin; had de detector toen al bestaan, dan waren er ongeveer 750 misschien ten onrechte aangemerkt als deels door AI geschreven. OpenAI bracht in januari 2023 een eigen detector uit. Op een verzameling Engelse teksten om mee te testen herkende die volgens het bedrijf 26% van de tekst van AI als waarschijnlijk door AI geschreven, en zag hij 9% van de tekst van mensen ten onrechte aan voor tekst van AI. In juli 2023 meldde TechCrunch dat OpenAI hem had stopgezet, omdat zijn accuracy te laag was.

    De vorige les zocht een ondertekend label. De meeste tekst heeft er geen, en dan blijft alleen de tekst zelf over. Deze les bouwt een detector die vraagt hoe verrast een model is door een tekst, probeert hem uit op antwoorden die Qwen3 en mensen op dezelfde vragen schreven, en bekijkt zijn fouten aan twee kanten.

    **Een voorbeeld uit het dagelijks leven.** Een winkelmedewerker die bankbiljetten op het gevoel controleert, vangt veel valse. Af en toe voelt een versleten echt biljet verkeerd en wordt het geweigerd, en een goede vervalsing glipt erdoor.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Zeven AI-detectors lazen 91 essays die mensen schreven voor de TOEFL, een toets Engels voor wie het als tweede taal leerde. Welk deel van die essays noemden de detectors gemiddeld door AI geschreven?",
        options={
            "few": "Ongeveer 5%",
            "right": "Ongeveer 61%",
            "quarter": "Ongeveer een kwart",
        },
        reveal="**Ongeveer 61%.** Liang et al. (2023) haalden de essays van een Chinees onderwijsforum. Volgens hen deden de detectors het bijna perfect bij essays van Amerikaanse scholieren uit de 8th grade, maar zagen ze meer dan de helft van de TOEFL-essays aan voor door AI gemaakt: gemiddeld een false positive rate van 61,22%. *Fouten tegen mensen* in *Het idee* zegt waarom.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Hoe verrast is het model?

    Een large language model (LLM) zoals Qwen3 leest een tekst token voor token, een woord of een deel van een woord, en geeft elk mogelijk volgend token een kans, gegeven de tokens ervoor. Voor elk token van een tekst is de **log-kans** de natuurlijke log van de kans die het model het gaf: nooit boven 0, en hoe lager, hoe minder het model dat token verwachtte. Min de log-kans is de verrassing van het token. De gemiddelde verrassing over alle tokens is de gemiddelde NLL van de tekst, zoals in [Hoe goed is een language model?](/course/open/lm-nll?lang=nl), waar de voorspellingen tekens waren.

    De paper over Binoculars, een detector uit 2024 van Hans et al., noemt het getal dat deze les uitrekent de log-perplexity: de gemiddelde negative log-likelihood van alle tokens van een tekst. Perplexity zelf is e tot de macht van de gemiddelde NLL. Waarom zou dat tekst van AI herkennen? Volgens de auteurs is de log-perplexity ook de loss waarmee generatieve LLM's getraind worden, dus vinden modellen hun eigen output waarschijnlijk weinig verrassend. Mensen verschillen van machines, en daarom vindt een LLM tekst van mensen verrassender. Volgens Mitchell et al. (2023) is de gangbaarste manier om tekst van een machine zero-shot te herkennen: de gemiddelde log-kans per token van de tekst nemen, en die met een drempel vergelijken. Zero-shot betekent zonder eerst een detector op voorbeelden te trainen. **AI text detection** is tekst die AI schreef onderscheiden van tekst die mensen schreven; deze les doet dat aan de tekst alleen.

    ### De run van de course

    De course gaf Qwen3-0.6B, het chat model van de course, 20 vragen die mensen aan het OpenAssistant-project stelden, zoals *Why is the sky blue?*. Qwen3 beantwoordde elke vraag in hooguit 256 tokens, en koos elk volgend token bij toeval uit zijn kansen, zoals de makers aanraden. Bij elke vraag hoort ook het antwoord dat een vrijwilliger in 2023 voor OpenAssistant schreef, en dat andere vrijwilligers het beste vonden. De course neemt die antwoorden als geschreven door mensen, zoals het project ze verzamelde; dat achteraf controleren is precies wat deze les moeilijk blijkt te maken. Daarna scoorde dezelfde Qwen3 alle 40 antwoorden: de log-kans van elk token, gegeven de tokens ervoor. Het eerste token van een antwoord krijgt geen score, want er komt niets voor.

    Los gescoord, zonder hun vraag, komen de antwoorden van mensen gemiddeld op 2,34, die van Qwen3 op 1,53. Maar ze overlappen: het laagste antwoord van een mens scoort 1,34, het hoogste van Qwen3 3,20. De antwoorden van Qwen3 kloppen ook niet altijd: het laat Socrates sterven in 401 voor Christus, in plaats van 399. Daar kijkt de score niet naar. De drie kortste antwoorden van Qwen3, van 20, 31 en 40 tokens, scoren 2,51, 3,20 en 2,41: over een paar tokens weegt één verrassend woord zwaar.

    De course scoorde elk antwoord ook na zijn vraag, in de chat template van Qwen3, zoals Qwen3 het las terwijl het schreef. Dan scoort elk antwoord van Qwen3 onder 0,74, en elk antwoord van een mens boven 1,28: een perfecte scheiding. Dat is het makkelijkste geval dat er is: de detector is precies het model dat de antwoorden schreef, en hij weet wat er gevraagd werd. De auteurs van Binoculars geven een voorbeeld. De prompt '1, 2, 3,' kan de heel voorspelbare voortzetting '4, 5, 6.' opleveren, terwijl een vraag om een paar zinnen over een capibara die astrofysicus is, een antwoord geeft dat verrassender lijkt. Zonder de prompt lijkt het herkennen van tekst van een LLM daarom moeilijk, en faalt een naïeve detector op basis van perplexity, schrijven ze. Een detector op school ziet het essay, niet de prompt, en weet niet welk model het schreef.
    """)
    return


@app.cell(hide_code=True)
def run_cell():
    mo.Html(C.run_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_mistakes():
    mo.md(r"""
    ### Een drempel en twee soorten fouten

    Een detector moet kiezen. De detector van deze les noemt een tekst AI als zijn gemiddelde NLL onder een **drempel** ligt, een getal dat vooraf gekozen is. Een tekst van een mens die hij AI noemt, is een **false positive**; de false positive rate is het deel van alle teksten van mensen dat hij AI noemt. Een tekst van AI die hij doorlaat, is een **false negative**. Wie de drempel verplaatst, ruilt de ene fout in voor de andere. Volgens Sadasivan et al. (2025) worden type-II-fouten talrijker naarmate een detector beter bestand is tegen type-I-fouten. Type-I-fouten zijn false positives, type-II-fouten false negatives. Bij de 40 losse antwoorden van de run van de course maakt geen enkele drempel minder dan 8 fouten.

    ### Fouten tegen mensen

    Een tekst die een model vaak gezien heeft, kan voor dat model voorspelbaar zijn, wie hem ook schreef. De auteurs van Binoculars scoorden 11 bekende documenten, waaronder de Amerikaanse grondwet; 3 daarvan vielen aan de machinekant van hun drempel. De course scoorde vijf bekende teksten met Qwen3, die van [Wat een model onthoudt](/course/open/rai-memorization?lang=nl). Het begin van *A Tale of Two Cities* komt gemiddeld op 1,62, de preambule van de Amerikaanse grondwet op 1,78: lager dan 19 en 16 van de 20 antwoorden van mensen. De andere drie komen op 3,03 tot 3,77, boven minstens 17 ervan: niet elke bekende tekst is voor een model voorspelbaar.

    Eenvoudige woorden zijn ook voorspelbaar. Liang et al. (2023) legden 91 TOEFL-essays en 88 essays van Amerikaanse scholieren uit de 8th grade, kinderen van ongeveer 13, voor aan zeven detectors die volgens hen veel gebruikt worden. Over de Amerikaanse essays zaten de detectors bijna altijd goed, maar van de TOEFL-essays noemden ze er gemiddeld 61,22% door AI geschreven. Alle zeven wezen 18 van de 91 TOEFL-essays (19,78%) aan als AI, en 89 van de 91 (97,80%) werden door minstens één detector aangewezen. Die 18 essays hadden een duidelijk lagere perplexity dan de andere. Toen ChatGPT de woorden van de TOEFL-essays rijker maakte, daalde de false positive rate van 61,22% naar 11,77%; toen het de woorden van de Amerikaanse essays eenvoudiger maakte, steeg die van 5,19% naar 56,65%. Hun advies: wees voorzichtig met een lage perplexity als teken van tekst van AI. Ook raden ze sterk af om zulke detectors te gebruiken bij het beoordelen van werk of in het onderwijs, vooral bij het werk van mensen voor wie Engels niet de moedertaal is. Ze noemen zelf een beperking: hun steekproeven zijn vrij klein.

    ### Gemiste tekst van AI

    Herschrijven verhoogt de verrassing. In dezelfde paper liet een tweede prompt ChatGPT zijn eigen essays voor de universiteit bewerken; daarna daalde het deel van 31 zulke essays dat herkend werd van 100% naar 13%. Sadasivan et al. lieten een tweede model tekst met een watermerk keer op keer parafraseren, in andere woorden herschrijven. Daardoor kon het deel dat herkend werd, bij 1% false positives, dalen van 99,3% naar 9,7%. Bij zero-shot detectors die de kansen van het model gebruiken, daalde de ROC AUC na parafraseren van 96,5% naar 25,2%. De ROC AUC geeft aan hoe goed de getallen van een detector de twee soorten tekst scheiden, over elke mogelijke drempel: 100% is een perfecte scheiding, en 50% is niet beter dan gokken. En mensen? Volgens Mitchell et al. doen mensen het maar iets beter dan toeval als ze tekst van een machine moeten onderscheiden van tekst van een mens.

    ### Betere detectors

    DetectGPT, van Mitchell et al., gaat ervan uit dat tekst die een LLM schreef, meestal ligt waar de log-kans van het model negatief gekromd is. Eenvoudiger gezegd: tekst van het model ligt bij een top van zijn log-kans. Kleine herschrijvingen ervan verlagen de log-kans meestal, terwijl ze bij tekst van mensen beide kanten op gaan. Bij nepnieuws dat GPT-NeoX schreef, een model van 20 miljard parameters, verbetert het de herkenning van een ROC AUC van 0,81 voor de sterkste eerdere zero-shot methode naar 0,95. Binoculars deelt de log-perplexity door hoe verrast het ene model is door wat een tweede model voorspelt. Op de eigen tests van de auteurs herkent het meer dan 90% van de teksten van ChatGPT (en andere LLM's) bij een false positive rate van 0,01%, zonder op data van ChatGPT getraind te zijn. Allebei hebben ze de kansen van een model nodig. Mitchell et al. noemen dat zelf een beperking: zulke methoden gaan ervan uit dat je de log-kansen van het model in kwestie kunt uitrekenen, een white-box-aanname. White-box betekent dat je in het model kunt kijken.

    ### Hoe vaak klopt een beschuldiging?

    Een false positive rate van 1% klinkt klein, maar welk deel van de beschuldigingen van een detector klopt, hangt ook af van hoe vaak tekst van AI voorkomt waar hij gebruikt wordt, de base rate. Neem 10.000 essays, waarvan 1 op de 10 van AI, en een detector die 90% daarvan vangt en 1% van de essays van mensen aanwijst. Van de 1000 essays van AI wijst hij er 900 aan; van de 9000 van mensen 90. Van zijn 990 beschuldigingen kloppen er 900: ongeveer 91%. Als tekst van AI zeldzamer is, heeft dezelfde detector minder vaak gelijk, zoals oefening 3 uitrekent.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **AI text detection** | tekst die AI schreef onderscheiden van tekst die mensen schreven |
        | **log-kans** | de natuurlijke log van de kans die een model een token geeft, gegeven de tokens ervoor; nooit boven 0 |
        | gemiddelde NLL, log-perplexity | het gemiddelde van min de log-kansen van de tokens van een tekst: de gemiddelde verrassing per token |
        | perplexity | e tot de macht van de gemiddelde NLL |
        | **drempel** | het getal waarmee een detector een score vergelijkt; hier betekent eronder AI |
        | **false positive** | een tekst van een mens die de detector AI noemt; de false positive rate is hun deel van alle teksten van mensen |
        | **false negative** | een tekst van AI die de detector doorlaat |
        | ROC AUC | hoe goed de getallen van een detector twee soorten tekst scheiden over elke drempel: 100% is een perfecte scheiding, 50% gokken |
        | `C.HUMAN`, `C.AI` | de gemiddelde NLL van de 20 antwoorden van mensen, en van de 20 van Qwen3, elk los gescoord |
        | `C.SKY_HUMAN`, `C.SKY_AI` | beide antwoorden op *Why is the sky blue?* zoals Qwen3 ze scoorde: paren `[token, logprob]` |
        """),
        L.watch_out("Een score van een detector toont nooit aan dat een tekst door AI geschreven is: laag betekent voorspelbaar voor dat model, niet geschreven door dat model. Beschuldig nooit iemand alleen op het woord van een detector; Liang et al. raden zulke detectors af om het werk van studenten te beoordelen, en Vanderbilt zette de detector van Turnitin uit."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Laat een model de tekst scoren: de log-kans van elk token, gegeven de tokens ervoor.
    2. Neem het gemiddelde van de verrassing, min de log-kans: laag betekent dat de tekst voor dat model voorspelbaar is.
    3. Kies een drempel, en tel beide fouten: aangewezen teksten van mensen, en doorgelaten tekst van AI.
    4. Vraag hoe vaak tekst van AI voorkomt waar de detector gebruikt wordt: is dat zelden, dan is een groot deel van de beschuldigingen fout.
    5. Zie een score als een reden om beter te kijken: hij toont niets aan.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel toont de eerste zes gescoorde tokens van beide antwoorden op *Why is the sky blue?*, elk met zijn verrassing, min zijn log-kans. `for token, logprob in C.SKY_HUMAN[:6]:` pakt elk van de eerste zes paren uit in zijn token en zijn log-kans; een token begint vaak met een spatie.
    """)
    return


@app.cell
def example_tokens():
    for token, logprob in C.SKY_HUMAN[:6]:   # het antwoord dat een mens schreef
        print("person:", token, round(-logprob, 2))
    for token, logprob in C.SKY_AI[:6]:      # het antwoord dat Qwen3 zelf schreef
        print("Qwen3: ", token, round(-logprob, 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.vstack([
        mo.md(r"""
        Beide antwoorden beginnen met dezelfde vier tokens, *The sky appears blue*, dus hun eerste drie verrassingen zijn gelijk: " sky" verrast Qwen3 het meest, met alleen "The" ervoor. Dan schrijft de mens " to us because", en Qwen3 zijn eigen " because of the", wat het veel minder verrast. Het plaatje toont de eerste 16 tokens van beide.
        """),
        mo.Html(C.tokens_picture(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_average():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. average_nll: de gemiddelde verrassing

        `C.SKY_HUMAN` bevat het antwoord van een mens op *Why is the sky blue?* zoals Qwen3 het scoorde: een lijst van paren `[token, logprob]`, zonder het eerste token. Schrijf `average_nll(pairs)`: de gemiddelde NLL van de lijst `pairs`, de gemiddelde verrassing per token, waarbij de verrassing van een token min zijn log-kans is. `for token, logprob in pairs:` pakt elk paar uit; tel de verrassingen op in `total`. Vervang de `...`. De check roept `average_nll(SKY_HUMAN)` aan, en daarna andere antwoorden en eigen lijsten.
        """),
        L.task("Tel de verrassing van elk token op, en geef het gemiddelde terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_average():
    box_average = L.code_box("average-nll", lines=5)
    box_average
    return (box_average,)


@app.cell(hide_code=True)
def check_average(box_average):
    result_average = L.check("average-nll", box_average.value)
    result_average
    return (result_average,)


@app.cell(hide_code=True)
def hints_average():
    L.hints(
        "average-nll",
        [
            "*Hoe verrast is het model?* in *Het idee* zegt hoe de verrassing uit een log-kans volgt, en wat de gemiddelde NLL is.",
            "`total` begint vóór de loop op 0; in de loop gaat de verrassing van elk token erin; na de loop geeft de functie het gemiddelde terug van wat ze optelde.",
            "`    return total / ___`",
        ],
        why="`total = 0`, dan `for token, logprob in pairs:` met `total = total - logprob` eronder, en `return total / len(pairs)`. `average_nll(SKY_HUMAN)` geeft ongeveer 2,286.",
    )
    return


@app.cell(hide_code=True)
def text_after_average():
    mo.md(r"""
    Over zijn 118 gescoorde tokens komt het antwoord van de mens gemiddeld op 2,29; over zijn 174 komt het eigen antwoord van Qwen3 op 1,01. Bij Socrates is het andersom: het antwoord van de mens komt op 1,64, en dat van Qwen3, één zin van 20 tokens, op 2,51. Eén getal per tekst; de detector moet ergens een grens trekken.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_flagged():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. flagged_share: hoeveel wijst de detector aan?

        De detector van deze les noemt een tekst AI als zijn gemiddelde NLL onder de drempel ligt. `C.HUMAN` bevat de gemiddelde NLL van de 20 antwoorden van mensen, `C.AI` die van de 20 antwoorden van Qwen3. Schrijf `flagged_share(scores, threshold)`: het deel van de getallen in de lijst `scores` dat onder `threshold` ligt, een getal van 0 tot 1. Tel ze in `flagged`, met een `for`-loop die elk getal van `scores` als `score` neemt. Vervang de `...`. De check roept `flagged_share(HUMAN, 1.75)` aan, en daarna `AI` en eigen lijsten.
        """),
        L.task("Tel de scores onder de drempel, en geef hun deel van alle scores terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_flagged():
    box_flagged = L.code_box("flagged-share", lines=6)
    box_flagged
    return (box_flagged,)


@app.cell(hide_code=True)
def check_flagged(box_flagged):
    result_flagged = L.check("flagged-share", box_flagged.value)
    result_flagged
    return (result_flagged,)


@app.cell(hide_code=True)
def hints_flagged():
    L.hints(
        "flagged-share",
        [
            "*Een drempel en twee soorten fouten* in *Het idee* zegt welke teksten de detector AI noemt.",
            "`flagged` begint op 0; in de loop beslist een `if` of er 1 bij komt; na de loop geeft de functie `flagged` terug als deel van alle scores.",
            "`        if score ___ threshold:`",
        ],
        why="`flagged = 0`, `for score in scores:`, dan `if score < threshold:` met `flagged = flagged + 1` eronder, en `return flagged / len(scores)`. `flagged_share(HUMAN, 1.75)` geeft 0,15: 3 van de 20 antwoorden van mensen.",
    )
    return


@app.cell(hide_code=True)
def text_after_flagged():
    mo.md(r"""
    Bij 1,75 is `flagged_share(AI, 1.75)` gelijk aan 0,75, dus een kwart van de antwoorden van Qwen3 komt erdoor, terwijl 3 van de 20 antwoorden van mensen aangewezen worden. Een daarvan is het antwoord van een mens over Socrates, terwijl het antwoord van Qwen3 op dezelfde vraag erdoor komt: bij één vraag zit de detector er twee keer naast, aan elke kant één keer.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: verplaats de drempel

    Verplaats de drempel, en kijk wat de detector AI noemt: de antwoorden van mensen, de antwoorden van Qwen3, en de vijf bekende teksten. Met het vakje aangevinkt scoort de detector elk antwoord na zijn vraag, zoals Qwen3 het las terwijl het schreef.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_threshold = mo.ui.slider(start=0.5, stop=3.5, step=0.05, value=C.THRESHOLD, label="Drempel", show_value=True)
    demo_question = mo.ui.checkbox(value=False, label="De detector kent de vraag")
    mo.hstack([demo_threshold, demo_question], justify="start", gap=2)
    return demo_question, demo_threshold


@app.cell(hide_code=True)
def demo_look(demo_question, demo_threshold):
    mo.Html(C.demo_picture(demo_threshold.value, demo_question.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_share():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. share_right: hoe vaak klopt een beschuldiging?

        Schrijf `share_right(caught, false_positive, share_ai)`: van de teksten die een detector aanwijst, het deel dat echt van AI is. `caught` is het deel van de teksten van AI dat hij aanwijst, `false_positive` het deel van de teksten van mensen dat hij aanwijst, en `share_ai` het deel van alle teksten dat van AI is; alle drie zijn het getallen van 0 tot 1. Reken zoals in *Hoe vaak klopt een beschuldiging?* in *Het idee* `right` uit, het deel van alle teksten dat van AI is en aangewezen wordt, en `wrong`, het deel van alle teksten dat van mensen is en aangewezen wordt. Vervang de `...`. De check roept `share_right(0.9, 0.01, 0.01)` aan: een detector die 90% van de teksten van AI vangt en 1% van de teksten van mensen aanwijst, terwijl 1 tekst op de 100 van AI is. Daarna probeert hij andere getallen.
        """),
        L.task("Reken right en wrong uit als delen van alle teksten, en geef het deel van right onder de aangewezen teksten terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_share():
    box_share = L.code_box("share-right", lines=4)
    box_share
    return (box_share,)


@app.cell(hide_code=True)
def check_share(box_share):
    result_share = L.check("share-right", box_share.value)
    result_share
    return (result_share,)


@app.cell(hide_code=True)
def hints_share():
    L.hints(
        "share-right",
        [
            "*Hoe vaak klopt een beschuldiging?* in *Het idee* rekent het uit voor 10.000 essays; hier werk je met delen van alle teksten in plaats van aantallen essays.",
            "`right` is `caught` maal `share_ai`. `wrong` is `false_positive` maal het deel van alle teksten dat mensen schreven. De laatste regel geeft `right` terug als deel van die twee samen.",
            "`    wrong = false_positive * ___`",
        ],
        why="`right = caught * share_ai`, `wrong = false_positive * (1 - share_ai)` en `return right / (right + wrong)`. `share_right(0.9, 0.01, 0.01)` geeft ongeveer 0,476: 90 van de 189 teksten die hij onder 10.000 aanwijst.",
    )
    return


@app.cell(hide_code=True)
def text_after_share():
    mo.md(r"""
    Van ongeveer 91% in *Het idee* naar ongeveer 48% hier: alleen hoe vaak tekst van AI voorkomt, veranderde. Met de getallen die OpenAI voor zijn detector gaf, 26% van de tekst van AI gevangen en 9% van de tekst van mensen aangewezen, en 1 tekst op de 10 van AI, geeft `share_right(0.26, 0.09, 0.1)` ongeveer 0,24: drie van elke vier beschuldigingen zouden fout zijn.
    """)
    return


@app.cell(hide_code=True)
def text_live():
    mo.md(r"""
    ### Probeer het op een eigen tekst

    Schrijf of plak een tekst, en Qwen3 scoort hem zoals in de run van de course: elk token met zijn verrassing, en het gemiddelde, tegen de drempel 1,75. Probeer een zin die je zelf schreef, een bekende regel, of dezelfde zin in eenvoudigere of zeldzamere woorden. Vergelijk teksten in dezelfde taal met elkaar: een model kent niet elke taal even goed.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Qwen3-0.6B downloaden (1,5 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    chat.download_or_stop(get_button, {
        "ask": "*Het LLM Qwen3-0.6B staat nog niet op je computer. De knop downloadt het één keer van Hugging Face, `Qwen/Qwen3-0.6B` in een vaste versie, 1,5 GB, onder de Apache-2.0-licentie, met de tokenizer van Qwen3-0.6B-Base. De rest van de les werkt zonder: die leest de scores die de course opnam.*",
        "busy": "**Downloaden:** {done} van {total} MB …",
        "no_net": "*Het downloaden lukte niet: Hugging Face was niet bereikbaar, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*Het downloaden lukte niet: het bestand dat binnenkwam is niet het vastgezette bestand van de course, dus de course gooide het weg. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*Het downloaden lukte niet: er is geen ruimte op je schijf voor 1,5 GB. Maak ruimte, en klik opnieuw.*",
        "failed": "*Het downloaden lukte niet ({why}). Klik opnieuw.*",
    })
    mo.md("*Qwen3-0.6B is gedownload.*")
    return


@app.cell(hide_code=True)
def live_controls():
    live_text = mo.ui.text_area(value=C.DEMO_TEXT, label="Jouw tekst", full_width=True, rows=4)
    score_button = mo.ui.run_button(label="Laat Qwen3 scoren")
    mo.vstack([live_text, score_button])
    return live_text, score_button


@app.cell(hide_code=True)
def live_look(live_text, score_button):
    if not chat.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif not score_button.value:
        _out = mo.md("*Druk op de knop om Qwen3 de tekst te laten scoren.*")
    elif not live_text.value.strip():
        _out = mo.md("*Schrijf eerst een tekst in het vak.*")
    else:
        mo.output.replace(mo.md("*Qwen3 leest je tekst …*"))
        _out = mo.Html(C.live_picture(C.live_pairs(live_text.value.strip()), L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_exercise_toefl():
    mo.md(r"""
    ### Oefening 4. De TOEFL-essays

    De zeven detectors van Liang et al. noemden gemiddeld 61,22% van de 91 TOEFL-essays door AI geschreven, en bijna geen van de 88 essays van Amerikaanse scholieren uit de 8th grade.
    """)
    return


@app.cell(hide_code=True)
def choice_toefl_cell():
    choice_toefl = L.choice(
        "toefl",
        question="Wat verklaart dat het best?",
        options={
            "copied": "Veel ervan waren deels met ChatGPT geschreven",
            "right": "Hun eenvoudigere woorden maken ze minder verrassend voor een model",
            "unseen": "De detectors hadden nooit essays gezien van mensen die Engels leren",
        },
        correct="right",
        why={
            "right": "De essays die alle zeven detectors aanwezen, hadden een duidelijk lagere perplexity. Met rijkere woorden daalde de false positive rate van 61,22% naar 11,77%; met eenvoudigere woorden steeg die van de Amerikaanse essays van 5,19% naar 56,65%. Laag betekent voorspelbaar, niet geschreven door AI.",
            "copied": "Toen ChatGPT de woorden van de TOEFL-essays rijker maakte, wezen de detectors er veel minder aan, van 61,22% naar 11,77%, niet meer. Ze reageren op hoe voorspelbaar de woorden zijn. Zie *Fouten tegen mensen* in *Het idee*.",
            "unseen": "Liang et al. wijzen naar de woorden: de essays die alle zeven aanwezen, hadden een lagere perplexity, en met eenvoudigere woorden leken ook de Amerikaanse essays op AI, van 5,19% naar 56,65%. Zie *Fouten tegen mensen* in *Het idee*.",
        },
    )
    choice_toefl
    return (choice_toefl,)


@app.cell(hide_code=True)
def text_exercise_rewrite():
    mo.md(r"""
    ### Oefening 5. Herschreven

    Een student laat een LLM een essay schrijven, en vraagt het daarna om het essay te herschrijven met zeldzamere, meer gevarieerde woorden.
    """)
    return


@app.cell(hide_code=True)
def choice_rewrite_cell():
    choice_rewrite = L.choice(
        "rewrite",
        question="Wat gebeurt er met de detector van deze les?",
        options={
            "down": "De gemiddelde NLL daalt, dus de detector wijst het zekerder aan",
            "right": "De gemiddelde NLL stijgt, en de detector mist het eerder",
            "same": "Niets: de detector herkent de ideeën van een LLM, niet zijn woorden",
        },
        correct="right",
        why={
            "right": "Zeldzamere woorden verrassen het model meer, dus de score schuift op naar die van tekst van mensen. Bij Liang et al. liet een prompt die ChatGPT zijn eigen essays liet bewerken, het deel van 31 essays van ChatGPT dat herkend werd, dalen van 100% naar 13%.",
            "down": "Zeldzamere woorden zijn minder voorspelbaar, dus ze verhogen de verrassing in plaats van die te verlagen. Daarom wezen de detectors met rijkere woorden minder TOEFL-essays aan. Zie *Gemiste tekst van AI* in *Het idee*.",
            "same": "De detector ziet alleen de log-kansen van de tokens, niet wat de tekst betekent. Herschrijven met andere woorden verandert die. Zie *Gemiste tekst van AI* in *Het idee*.",
        },
    )
    choice_rewrite
    return (choice_rewrite,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Waar komt dit plaatje vandaan?](/course/open/rai-content-provenance?lang=nl) zei een ondertekend label wie voor een plaatje instaat. Zonder label keek deze les naar de tekst alleen, met de gemiddelde NLL van [Hoe goed is een language model?](/course/open/lm-nll?lang=nl): tekst die een model schreef, verrast het meestal minder. Maar een score aan één kant van een drempel toont niets aan. Sommige bekende teksten en eenvoudige woorden komen aan de kant van AI terecht, herschreven tekst van AI aan de andere kant, en hoe vaak een beschuldiging klopt, hangt af van hoe vaak tekst van AI voorkomt. Dit is de laatste les van de module over security. De volgende module, over AI-veiligheid, begint met *Soorten risico's van AI*.

    **Een voorbeeld uit het dagelijks leven.** Een rookmelder die afgaat zodra er een toastje aanbrandt, mist geen brand, en een die nooit op toast reageert, mist misschien een smeulend vuur. Waar je hem afstelt, is een keuze tussen de twee fouten; en in een keuken waar zelden brand is, gaan de meeste alarmen over toast.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_rewrite, choice_toefl, result_average, result_flagged, result_share):
    L.wrap_up(result_average, result_flagged, result_share, choice_toefl, choice_rewrite)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
