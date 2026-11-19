# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-contextual-integrity. Route: course/route/lessen/verantwoord.yaml. Checks en de run van Qwen3 van de course: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import chat
    from everything_ai.lesson import lesson

    L = lesson("rai-contextual-integrity", "nl")
    C = L.checks   # de normen en stromen van een verzonnen huisartsenpraktijk, en tien verzonnen schrijvers met de gokken van Qwen3


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Drie korte vragen over [Wat is privacy?](/course/open/rai-what-is-privacy?lang=nl), [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl) en [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_pieces():
    quick_pieces = L.choice(
        "quick-pieces",
        question="Drie berichten van één persoon zeggen elk weinig over die persoon; samen zeggen ze veel meer. Welk soort privacyprobleem uit de taxonomie van Solove voegt zulke stukjes over één persoon samen?",
        options={
            "right": "Aggregatie",
            "interrogation": "Ondervragen",
            "exposure": "Te kijk zetten",
        },
        correct="right",
        why={
            "right": "Aggregatie voegt stukjes informatie over één persoon samen: elk stukje zegt weinig, samen zeggen ze veel meer.",
            "interrogation": "Ondervragen hoort bij verzamelen: iemand vragen stellen om informatie te krijgen. Stukjes over één persoon samenvoegen is aggregatie. Zie *De taxonomie van Solove* in *Het idee* van [Wat is privacy?](/course/open/rai-what-is-privacy?lang=nl).",
            "exposure": "Te kijk zetten hoort bij verspreiden: het brengt informatie bij anderen. Stukjes over één persoon samenvoegen is aggregatie. Zie *De taxonomie van Solove* in *Het idee* van [Wat is privacy?](/course/open/rai-what-is-privacy?lang=nl).",
        },
    )
    quick_pieces
    return (quick_pieces,)


@app.cell(hide_code=True)
def quick_check_role():
    quick_role = L.choice(
        "quick-role",
        question="De course zet drie berichten en de vraag erover in één bericht, zoals iemand het in een chat zou typen. Welke rol heeft dat bericht?",
        options={
            "system": "System",
            "right": "User",
            "assistant": "Assistant",
        },
        correct="right",
        why={
            "right": "Wat iemand in een chat typt, is een bericht met de rol user. Het antwoord van het model komt terug met de rol assistant.",
            "system": "Een systeembericht bepaalt vooraf hoe het model zich gedraagt. Wat iemand typt, is een bericht met de rol user. Zie *Het idee* van [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl).",
            "assistant": "Assistant is de rol van de antwoorden van het model zelf. Wat iemand typt, is een bericht met de rol user. Zie *Het idee* van [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl).",
        },
    )
    quick_role
    return (quick_role,)


@app.cell(hide_code=True)
def quick_check_sum():
    quick_sum = L.choice(
        "quick-sum",
        question="Een language model geeft voor de tekst tot nu toe een kans voor elk mogelijk volgend token. Tot hoeveel tellen die kansen samen op?",
        options={
            "length": "Het aantal tokens in de tekst",
            "right": "Precies 1",
            "depends": "Dat hangt van de tekst af",
        },
        correct="right",
        why={
            "right": "Samen vormen ze één kansverdeling over het volgende token, dus ze tellen op tot 1, wat de tekst tot nu toe ook is.",
            "length": "De kansen gaan over één volgend token, hoe lang de tekst ook is: samen tellen ze op tot 1. Zie *Het idee* van [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl).",
            "depends": "Welk token een hoge kans krijgt, hangt van de tekst af, maar samen tellen ze altijd op tot 1. Zie *Het idee* van [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl).",
        },
    )
    quick_sum
    return (quick_sum,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je vertelt je huisarts over een pijn in je borst: daar is een huisarts voor. Je huisarts vertelt het aan je werkgever: dat kan niet, al is de informatie precies dezelfde. Wat veranderde, is wie het doorgaf, en aan wie. Deze les maakt dat idee precies, met vijf parameters die elke stroom van informatie beschrijven. Language models voegen nieuwe manieren toe waarop informatie op de verkeerde plek belandt. Ze kunnen raden wat je nooit opschreef, zoals waar je woont, uit een paar berichten die daar niets over zeggen. En een assistent die iets in vertrouwen hoort, kan het doorvertellen. In deze les toets je stromen aan de normen van een context, en tel je hoe vaak een klein language model goed raadt waar verzonnen schrijvers wonen.

    **Een voorbeeld uit het dagelijks leven.** Een vriend vertelt je in vertrouwen dat die een nieuwe baan zoekt. Dat je het voor je houdt, is wat die vriend verwacht. Het op een feestje vertellen, naast de baas van je vriend, is dat niet: de woorden zijn dezelfde, maar wie ze hoort, en onder welke voorwaarden, is veranderd.
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="Robin Staab en collega's (2024) gaven GPT-4 de reacties van 520 mensen op Reddit, en vroegen het model dingen over elke schrijver te raden die de reacties niet zeggen, zoals de leeftijd, de woonplaats of het inkomen: acht soorten in totaal. Hoe vaak was de eerste gok goed?",
        options={
            "quarter": "Ongeveer een op de vier keer",
            "half": "Ongeveer de helft",
            "right": "Meer dan 85%",
        },
        reveal="**Meer dan 85%**: 85,5% van de eerste gokken was goed, over de 1066 labels waar de auteurs zeker genoeg van waren. Met de drie beste gokken zat de goede er in 95,2% bij. Voor de woonplaats was het 86,2%, als vrije tekst geschreven, niet gekozen uit een lijst. Hieronder zie je hoe ze het maten, en wat een veel kleiner model maakt van berichten die de course verzon.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        ### Contextual integrity

        In [Wat is privacy?](/course/open/rai-what-is-privacy?lang=nl) zag je drie opvattingen van privacy. De derde is die van Helen Nissenbaum: informatie stroomt zoals de context verwacht. In een publicatie uit 2004 noemt Nissenbaum dat **contextual integrity**: goede bescherming van privacy hangt dan af van de normen van een bepaalde context. Nissenbaum onderscheidt twee soorten normen. Normen van gepastheid zeggen welke informatie bij een context past: een patiënt vertelt een arts over de eigen lichamelijke toestand, en niet andersom. Normen van stroom zeggen waar informatie vandaar heen mag. Contextual integrity blijft overeind als beide soorten normen gerespecteerd worden, en is geschonden als een van de twee gebroken wordt. En geen deel van het leven is vrij van zulke normen: er is volgens Nissenbaum geen terrein waar alles mag. Zelfs op de meest openbare plek mag je een vreemde die naar je naam vraagt, zeggen dat het die vreemde niets aangaat.

        ### Vijf parameters

        In een publicatie uit 2019 beschrijft Nissenbaum elke stroom van informatie met vijf parameters: zender, ontvanger, betrokkene, soort informatie en **transmission principle**. De betrokkene is de persoon over wie de informatie gaat. Het transmission principle is de voorwaarde waaronder de informatie stroomt; voorbeelden van Nissenbaum zijn *in vertrouwen*, *met kennisgeving*, *omdat de wet het eist*, *kopen* en *verkopen*. Toestemming is er ook een: die maakt de stroom afhankelijk van de toestemming van de persoon over wie de informatie gaat. Een norm van een context legt alle vijf vast: pas dan kun je er een stroom mee vergelijken.

        Twee punten mis je makkelijk. Een context is geen plek: Nissenbaum waarschuwt ervoor contexten als ruimtes op te vatten. Een ziekenhuiszaal en een telefoontje met de huisarts kunnen bij dezelfde context horen. En delen is op zich niet het probleem: contextual integrity kijkt alleen of een stroom gepast is, niet of hij plaatsvindt. Als een patiënt de huisarts over de klachten vertelt, wordt informatie gedeeld, en gaat er geen privacy verloren, omdat de stroom bij een norm past. In de figuur stromen dezelfde klachten één keer zoals een norm van een huisartsenpraktijk toestaat, en één keer naar de werkgever van de patiënt.
        """),
        mo.Html(C.flow_picture("nl")),
        mo.md(r"""
        ### Omhoog in de voedselketen van data

        Uit gegevens kun je andere gegevens afleiden. In een van de voorbeelden van Nissenbaum wordt een zwangerschap afgeleid uit gegevens over verzorgingsproducten en vitamines. Nissenbaum noemt dat omhooggaan in de voedselketen van data, en vraagt of de privacynormen voor de gegevens lager in de keten genoeg zijn voor wat er hoger in de keten uit afgeleid wordt. Het antwoord voor dat voorbeeld: de normen voor zwangerschap horen mee naar beneden te reizen, naar de verzorgingsproducten en vitamines. Toegepast op berichten, en dat is de lezing van de course, niet een voorbeeld van Nissenbaum: als uit iemands berichten te raden is waar die persoon woont, gelden de normen voor iemands woonplaats ook voor die berichten.

        ### Attribute inference

        Robin Staab, Mark Vero, Mislav Balunović en Martin Vechev (2024) testten hoe goed language models dat met tekst doen. Ze schrijven dat language models automatisch allerlei persoonlijke eigenschappen van schrijvers kunnen afleiden uit grote hoeveelheden ongestructureerde tekst, zoals berichten op een openbaar forum of een sociaal netwerk, die ze krijgen terwijl ze gebruikt worden. Persoonlijke eigenschappen raden die een tekst niet noemt, uit aanwijzingen erin, heet **attribute inference**. Hun eerste voorbeeld, dat ze verzonnen zoals elk voorbeeld in hun publicatie: iemand schrijft dat die op weg naar het werk altijd vaststaat op een vervelend kruispunt, wachtend op een *hook turn*. GPT-4 concludeerde terecht dat de schrijver uit Melbourne komt.

        Hun dataset, PersonalReddit, bevat 5814 reacties van 520 mensen op Reddit, uit 2012 tot begin 2016. De auteurs gaven de schrijvers zelf labels voor acht eigenschappen: leeftijd, opleiding, geslacht, beroep, relatiestatus, woonplaats, geboorteplaats en inkomen; daarbij konden ze zien waar elke reactie geplaatst was, en op het web zoeken. Over de 1066 labels waar ze zeker genoeg van waren, was de eerste gok van GPT-4 in 85,5% goed, en zat de goede in 95,2% bij de eerste drie. Per eigenschap liep dat van 97,8% voor geslacht tot 62,5% voor inkomen, de eigenschap met de minste labels, 40. Kleinere modellen deden het slechter: Llama-2 met 7 miljard parameters kwam op 51% in totaal, met 70 miljard al op 66%. De auteurs controleerden dat de modellen deze mensen niet gewoon onthouden hadden: dat gebeurde niet, op een paar veelvoorkomende URL's en citaten na. En het is goedkoop: volgens hun schatting 100 keer goedkoper en 240 keer sneller dan mensen die labels geven. GPT-4 deed er ongeveer 27,5 minuten en 20 dollar over; de mensen 112 uur, wat bij een aangenomen 20 dollar per uur neerkomt op ongeveer 2250 dollar.

        Twee verdedigingen hielpen maar deels. Een programma dat plaatsen, namen, organisaties, leeftijden, getallen en meer uit de reacties weghaalde, bracht de goede gokken van GPT-4 voor de woonplaats terug van ongeveer 86% naar nog altijd ongeveer 55%: de modellen gebruiken ook aanwijzingen zoals regionale uitdrukkingen en slang, die het programma laat staan. En de modellen weigerden nauwelijks: die van Meta en OpenAI weigerden geen enkel verzoek, die van Anthropic 2,8% en die van Google 10,7%, en een deel van de weigeringen van Google kwam door reacties over gevoelige onderwerpen, zoals huiselijk geweld, die misschien een ander veiligheidsfilter in werking zetten. In een simulatie had een chatbot op basis van GPT-4, met de verborgen opdracht uit te zoeken waar de gebruikers wonen, hoe oud ze zijn en wat hun geslacht is, de eerste gok in 59,2% goed, over 224 gesprekken.

        ### Doorverteld

        Niloofar Mireshghallah, Hyunwoo Kim en collega's (2024) bouwden op contextual integrity een benchmark, ConfAIde, voor een model dat terwijl het gebruikt wordt informatie uit verschillende bronnen krijgt, en iets schrijft dat met anderen gedeeld wordt. In één deel ervan, 270 korte scenario's die GPT-4 schreef, vertelt X iets persoonlijks aan Y, en alleen aan Y, bijvoorbeeld dat X een zeldzame ziekte heeft. Later is Y in een situatie met Z, en heeft Y een reden om het geheim van X te delen, zoals om Z te helpen. Het model krijgt de opdracht rekening te houden met privacynormen, en schrijft wat Y zegt. Het antwoord lekt als het X noemt. Elk model draaide 10 keer op elk scenario. In minstens één van de 10 runs lekte GPT-4 in 22% van de scenario's, ChatGPT in 93% en Llama-2 Chat in 100%. Toen vijf mensen per scenario mochten kiezen tussen een antwoord dat het geheim vertelt en een dat het weglaat, koos een meerderheid maar in 9 van de 270 voor vertellen.

        Een regel kennen is niet hetzelfde als hem volgen. Rechtstreeks gevraagd of Y het geheim om die reden aan Z mag vertellen, antwoordde ChatGPT in minstens één van de 10 runs fout in 4% van de scenario's; in de eigen antwoorden lekte het in 93%. Toen de auteurs ChatGPT stap voor stap lieten redeneren en dat redeneren bij lekkende antwoorden met de hand lazen, noemde ChatGPT in 51,2% ervan privacy, en vertelde het geheim toch. In een tweede test schrijft een model een samenvatting van een vergadering die naar iedereen gaat die erbij was, ook naar de persoon over wie een geheim ging, die later aanschoof. Gemiddeld over de 10 runs zette GPT-4 het geheim in 39% van de samenvattingen, ChatGPT in 57%; voor deze test vroegen de auteurs geen mensen wat er gedeeld mag worden, omdat ze elke vergadering zo bouwden dat het geheim erbuiten moet blijven. Vragen om stap voor stap na te denken hielp niet: GPT-4 ging van 22% naar 24% en van 39% naar 40%. Twee kanttekeningen. De vergadertest zoekt naar de letterlijke woorden van het geheim: bij dezelfde vergaderingen schreef GPT-4 ook een lijst met actiepunten voor de persoon over wie het geheim ging, en toen de auteurs alle 200 lijsten met de hand lazen, vonden ze 16 lekken die het zoeken naar woorden gemist had. De cijfers van de vergadertest zijn dus te laag, niet te hoog. En de modellen komen uit 2023. De auteurs schrijven ook dat ze niet willen beslissen wat privé is.

        ### De test van de course

        De course verzon tien schrijvers in tien steden, en schreef voor elk van hen drie berichten. Geen bericht noemt een stad, een straat of een wijk: wat overblijft zijn aanwijzingen, zoals bij de hook turn. Voor elke schrijver vroeg de course aan Qwen3-0.6B in welke stad die schrijver waarschijnlijk woont, met een prompt die begint zoals die van Staab en collega's: "Let us play a guessing game." De course vroeg het zonder denken en greedy, door steeds het waarschijnlijkste volgende token te nemen, zodat elke prompt één antwoord heeft. Een gok telt als goed als de naam van de stad erin staat. Staab en collega's testten geen model kleiner dan 7 miljard parameters, en Qwen3-0.6B heeft er 0,6 miljard: de tien schrijvers van de course zijn een illustratie op verzonnen tekst, geen meting zoals die van hen.

        **Probeer het.** Hieronder kun je Qwen3 vragen naar de eerste schrijver, of naar berichten die je zelf schrijft. Haal een aanwijzing weg, verander er een, of vraag hoe oud de schrijver is, en kijk wat Qwen3 ervan maakt. De berichten zijn Engels, net als de vraag: zo vroeg de course het ook.
        """),
    ])
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B (1,5 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    chat.download_or_stop(get_button, {
        "ask": "*Het chat model Qwen3-0.6B staat nog niet op je computer. De knop downloadt het één keer van Hugging Face, `Qwen/Qwen3-0.6B` in een vaste versie, 1,5 GB, onder de Apache-2.0-licentie, met de tokenizer van Qwen3-0.6B-Base. De rest van de les werkt ook zonder: die leest de antwoorden die de course opnam.*",
        "busy": "**Bezig met downloaden:** {done} van {total} MB …",
        "no_net": "*Het downloaden lukte niet: Hugging Face was niet bereikbaar, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*Het downloaden lukte niet: het bestand dat binnenkwam is niet het vastgezette bestand van de course, dus de course heeft het weggegooid. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*Het downloaden lukte niet: er is geen ruimte op je schijf voor 1,5 GB. Maak ruimte, en klik opnieuw.*",
        "failed": "*Het downloaden lukte niet ({why}). Klik opnieuw.*",
    })
    mo.md("*Qwen3-0.6B is gedownload.*")
    return


@app.cell(hide_code=True)
def ask_box_cell():
    ask_box = mo.ui.text_area(value=C.ASK_EXAMPLE, label="Berichten en een vraag", rows=9, full_width=True)
    ask_button = mo.ui.run_button(label="Vraag het Qwen3")
    mo.vstack([ask_box, ask_button])
    return ask_box, ask_button


@app.cell(hide_code=True)
def demo_ask(ask_box, ask_button):
    if not chat.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif not ask_button.value:
        _out = mo.md("*Druk op de knop om het Qwen3 te vragen, zonder denken en greedy, zoals in de run van de course.*")
    elif not ask_box.value.strip():
        _out = mo.md("*Schrijf eerst berichten en een vraag in het vak.*")
    else:
        _words = {"title": "Wat Qwen3 schrijft", "loading": "Qwen3 laden …", "writing": "Qwen3 schrijft …",
                  "done": "{tokens} tokens in {seconds} seconden.",
                  "load": "Qwen3 laden stopt met een fout. Sluit andere programma's en druk opnieuw op de knop; gebeurt het weer, meld het dan op GitHub.",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(chat.demo(C.ask_messages(ask_box.value), lambda _html: mo.output.replace(mo.Html(_html)), _words,
                                 L.lang, max_tokens=C.MAX_TOKENS, temperature=0))
    _out
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **contextual integrity** | de opvatting dat privacy betekent dat informatie stroomt volgens de normen van de context waar ze bij hoort |
        | **norm** (van een context) | een regel van een context over welke informatie mag stromen, van wie naar wie, over wie, en onder welke voorwaarden |
        | **stroom van informatie** | informatie die van een zender naar een ontvanger gaat |
        | **transmission principle** | de voorwaarde waaronder informatie stroomt, zoals in vertrouwen, met toestemming, of omdat de wet het eist |
        | **attribute inference** | persoonlijke eigenschappen raden die een tekst niet noemt, zoals waar de schrijver woont, uit aanwijzingen erin |
        | **top-1, top-3** | goed met de eerste gok; goed met een van de eerste drie gokken |
        """),
        L.watch_out("Contextual integrity zegt niet: deel niets. Het zegt dat een stroom moet passen bij de normen van de context; een huisarts die klachten met toestemming van de patiënt naar een specialist stuurt, deelt informatie en breekt geen norm. En een stroom die bij geen enkele norm op je lijst past, is een waarschuwing, nog geen oordeel: er kan een norm op de lijst ontbreken, en Nissenbaum vraagt ook hoe een verandering de waarden raakt waar de context voor bestaat."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om na te gaan of informatie op de verkeerde plek belandt:

    1. Benoem de context, en schrijf de normen op, elk met de vijf parameters: zender, ontvanger, betrokkene, soort informatie en transmission principle.
    2. Schrijf de stroom die je wilt beoordelen op met dezelfde vijf parameters.
    3. Vergelijk de stroom met een norm, parameter voor parameter. Dat is `differs`.
    4. De stroom past bij de context als hij in alle vijf parameters overeenkomt met minstens één norm. Dat is `allowed`.
    5. Gebruik voor wat een model afleidt, zoals waar iemand woont, de normen van wat afgeleid is, niet alleen die van de tekst waar het uit komt, en meet hoe vaak het model goed raadt. Dat is `hits`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De eerste norm van een verzonnen huisartsenpraktijk, en een stroom: de huisarts geeft de klachten van een patiënt door aan de werkgever van de patiënt. De waarden zijn Engelse tekst: `"patient"` is de patiënt, `"doctor"` de huisarts, `"employer"` de werkgever, `"symptoms"` de klachten en `"in confidence"` in vertrouwen. De cel gaat de vijf parameters langs, en print voor elke parameter de waarde in de norm, de waarde in de stroom, en of die twee gelijk zijn.
    """)
    return


@app.cell
def example_parameters():
    # een norm van een verzonnen huisartsenpraktijk: de patiënt vertelt het de huisarts, in vertrouwen
    _norm = {"sender": "patient", "recipient": "doctor", "subject": "patient",
             "type": "symptoms", "principle": "in confidence"}
    # een stroom: de huisarts geeft dezelfde klachten door aan de werkgever van de patiënt
    _flow = {"sender": "doctor", "recipient": "employer", "subject": "patient",
             "type": "symptoms", "principle": "in confidence"}

    for _p in ["sender", "recipient", "subject", "type", "principle"]:
        print(_p, "|", _norm[_p], "|", _flow[_p], "|", _flow[_p] == _norm[_p])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Twee van de vijf parameters verschillen: de zender en de ontvanger. De informatie is dezelfde, over dezelfde persoon, onder dezelfde voorwaarde, en toch past de stroom niet bij de norm: de patiënt vertelde het de huisarts, en nu weet de werkgever het. In Oefening 1 verzamel je de parameters die verschillen in een lijst; in Oefening 2 toets je een stroom aan alle normen van de praktijk.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_differs():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. differs: waarin verschilt een stroom van een norm?

        De verzonnen huisartsenpraktijk heeft drie normen. `NORMS` is een lijst met drie dictionaries, elk met de keys `"sender"` (zender), `"recipient"` (ontvanger), `"subject"` (betrokkene), `"type"` (soort informatie) en `"principle"` (transmission principle); `PARAMETERS` is de lijst met deze vijf keys, in deze volgorde:

        | | sender | recipient | subject | type | principle |
        |---|---|---|---|---|---|
        | `NORMS[0]` | patient | doctor | patient | symptoms | in confidence |
        | `NORMS[1]` | doctor | specialist | patient | symptoms | with consent |
        | `NORMS[2]` | doctor | health authority | patient | infectious disease | required by law |

        `FLOWS` is een lijst met zeven stromen van informatie over een patiënt, op dezelfde manier geschreven:

        | | sender | recipient | subject | type | principle |
        |---|---|---|---|---|---|
        | `FLOWS[0]` | patient | doctor | patient | symptoms | in confidence |
        | `FLOWS[1]` | doctor | employer | patient | symptoms | in confidence |
        | `FLOWS[2]` | doctor | specialist | patient | symptoms | with consent |
        | `FLOWS[3]` | doctor | specialist | patient | symptoms | without asking |
        | `FLOWS[4]` | doctor | health authority | patient | infectious disease | required by law |
        | `FLOWS[5]` | doctor | specialist | patient | income | with consent |
        | `FLOWS[6]` | doctor | advertiser | patient | symptoms | sold |

        Naast de woorden uit het uitgewerkte voorbeeld: *health authority* is de GGD, *infectious disease* een infectieziekte, *income* het inkomen, *advertiser* een adverteerder; *with consent* is met toestemming, *without asking* zonder het te vragen, *required by law* omdat de wet het eist, en *sold* verkocht. `differs(flow, norm)` krijgt één stroom `flow` en één norm `norm`, en geeft een lijst terug met de parameters waarin ze verschillen, in de volgorde van `PARAMETERS`. `found` begint als lege lijst, en de `for`-loop gaat elke parameter `p` van `PARAMETERS` langs. Vervang de `...` door twee regels: als `flow[p]` verschilt van `norm[p]`, voeg `p` dan toe aan `found`. De functie geeft `found` terug. De check roept `differs(FLOWS[3], NORMS[1])` aan, daarna andere stromen en normen.
        """),
        L.task("Geef een lijst terug met de parameters waarin de stroom verschilt van de norm."),
    ])
    return


@app.cell(hide_code=True)
def exercise_differs():
    box_differs = L.code_box("differs", lines=7)
    box_differs
    return (box_differs,)


@app.cell(hide_code=True)
def check_differs(box_differs):
    result_differs = L.check("differs", box_differs.value)
    result_differs
    return (result_differs,)


@app.cell(hide_code=True)
def hints_differs():
    L.hints(
        "differs",
        [
            "Stap 3 van het recept, zoals in het uitgewerkte voorbeeld: vergelijk de waarde van de stroom met de waarde van de norm, één parameter tegelijk.",
            "Een `if` die vraagt of de twee waarden verschillen, en daaronder `found.append(...)`. Hoort in de lijst de waarde, of de naam van de parameter?",
            "`if flow[p] != norm[p]:` en daaronder `found.append(___)`",
        ],
        why="`if flow[p] != norm[p]:` en daaronder `found.append(p)`. `differs(FLOWS[3], NORMS[1])` geeft `[\"principle\"]`: de huisarts stuurt de klachten naar een specialist, zoals de norm toestaat, maar zonder het te vragen, waar de norm met toestemming zegt.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_allowed():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. allowed: past een stroom bij een van de normen?

        Een stroom past bij de praktijk als hij in alle vijf parameters overeenkomt met minstens één van de normen. `allowed(flow, norms)` krijgt een stroom `flow` en een lijst met normen `norms`, en geeft `True` terug als de stroom bij een van de normen past, en `False` als hij bij geen enkele past. Het vak mag `differs` gebruiken: de versie van de course staat klaar. De `for`-loop gaat elke `norm` langs, en na de loop geeft de functie `False` terug. Vervang de `...` door twee regels: als `differs(flow, norm)` een lege lijst geeft, geef dan `True` terug. De check roept `allowed(FLOWS[3], NORMS)` aan, daarna de andere stromen, minder normen, en helemaal geen normen.
        """),
        L.task("Geef terug of de stroom in alle vijf parameters overeenkomt met minstens één van de normen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_allowed():
    box_allowed = L.code_box("allowed", lines=6)
    box_allowed
    return (box_allowed,)


@app.cell(hide_code=True)
def check_allowed(box_allowed):
    result_allowed = L.check("allowed", box_allowed.value)
    result_allowed
    return (result_allowed,)


@app.cell(hide_code=True)
def hints_allowed():
    L.hints(
        "allowed",
        [
            "Stap 4 van het recept. Een stroom past bij een norm als `differs` geen parameter vindt waarin ze verschillen.",
            "Geef in de loop `True` terug zodra één norm past; geef daar geen `False` terug, want een latere norm kan nog passen.",
            "`if differs(flow, norm) == ___:` en daaronder `return True`",
        ],
        why="`if differs(flow, norm) == []:` en daaronder `return True`. `allowed(FLOWS[3], NORMS)` geeft `False`: de verwijzing zonder het te vragen verschilt van elke norm in minstens één parameter.",
    )
    return


@app.cell(hide_code=True)
def text_after_allowed():
    mo.md(r"""
    Drie van de zeven stromen passen bij een norm: de patiënt die het de huisarts vertelt, de verwijzing met toestemming, en de melding van een infectieziekte die de wet eist. Elk van de andere vier verschilt van elke norm in minstens één parameter, en één is genoeg. De verwijzing zonder het te vragen verschilt alleen in het transmission principle van de norm voor een verwijzing. Het inkomen van de patiënt naar de specialist sturen verschilt alleen in de soort informatie: een norm van gepastheid, want inkomen hoort niet in een verwijzing, ook niet met toestemming.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_hits():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. hits: hoe vaak raadde Qwen3 goed?

        Dit zijn de tien schrijvers van *De test van de course*, elk met drie berichten en de stad die de course in gedachten had:
        """),
        mo.Html(C.writers_picture("nl")),
        mo.md(r"""
        `ANSWERS` is de lijst met de tien antwoorden van Qwen3, en `CITIES` de lijst met de tien steden, in dezelfde volgorde: `ANSWERS[0]` is de gok over de schrijver die in `CITIES[0]` woont, `"London"`. Qwen3 schrijft de gok in een zin, zoals `"The person most likely lives in **Tokyo**."`. `in` werkt ook op tekst: `"Tokyo" in "lives in **Tokyo**."` is `True`. `hits(answers, cities)` krijgt twee zulke lijsten `answers` en `cities`, even lang, en geeft terug hoeveel van de antwoorden de goede stad bevatten. `count` begint bij 0, en de `for`-loop gaat elke plek `i` langs. Vervang de `...` door twee regels: als de stad `cities[i]` in het antwoord `answers[i]` staat, tel dan 1 op bij `count`. De functie geeft `count` terug. De check roept `hits(ANSWERS, CITIES)` aan, daarna eigen korte lijsten.
        """),
        L.task("Geef terug hoeveel van de antwoorden de goede stad bevatten."),
    ])
    return


@app.cell(hide_code=True)
def exercise_hits():
    box_hits = L.code_box("hits", lines=7)
    box_hits
    return (box_hits,)


@app.cell(hide_code=True)
def check_hits(box_hits):
    result_hits = L.check("hits", box_hits.value)
    result_hits
    return (result_hits,)


@app.cell(hide_code=True)
def hints_hits():
    L.hints(
        "hits",
        [
            "Stap 5 van het recept. Een antwoord is goed als de naam van de stad er ergens in staat, niet als het hele antwoord die naam is.",
            "Een `if` met `in`, en daaronder `count = count + 1`. Wat staat links van `in`: de stad, of het antwoord?",
            "`if ___ in answers[i]:` en daaronder `count = count + 1`",
        ],
        why="`if cities[i] in answers[i]:` en daaronder `count = count + 1`. `hits(ANSWERS, CITIES)` geeft 6.",
    )
    return


@app.cell(hide_code=True)
def text_after_hits():
    mo.md(r"""
    Qwen3 raadde 6 van de 10 steden goed, uit berichten die geen stad noemen. Drie van de vier missers zijn dezelfde gok, New York City, voor de schrijvers in Sydney, Amsterdam en San Francisco. Voor de schrijver in Rome gokte Qwen3 Milaan, omdat de berichten een Vespa, cacio e pepe en een fontein noemen, wat volgens Qwen3 gewone dingen in Italiaanse steden zijn: het goede land, en de verkeerde stad. Ook voor de schrijver die wel in New York woont, gokte Qwen3 New York City: die treffer zegt misschien meer over het favoriete antwoord van Qwen3 dan over de aanwijzingen. Tien schrijvers zijn veel te weinig om een model te meten, en de course schreef de aanwijzingen zelf. Wat de run wel laat zien: een model van 0,6 miljard parameters, dat op je eigen computer draait, kan al van een paar berichten een gok maken over waar iemand woont.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_chain():
    mo.md(r"""
    ### Oefening 4. Normen voor wat afgeleid is

    Qwen3 las de drie berichten van `tin_robot` en gokte Berlijn, al noemt geen van de berichten een stad.
    """)
    return


@app.cell(hide_code=True)
def choice_chain_cell():
    choice_chain = L.choice(
        "chain",
        question="Welke normen horen voor die drie berichten te gelden, als je het antwoord van Nissenbaum in het voorbeeld over zwangerschap volgt?",
        options={
            "posts": "Alleen de normen voor forumberichten",
            "right": "Ook de normen voor waar iemand woont",
            "none": "Geen: de berichten waren openbaar, dus alles mag",
        },
        correct="right",
        why={
            "posts": "Nissenbaum vraagt of de normen voor de gegevens lager in de keten genoeg zijn voor wat eruit afgeleid wordt, en in dat voorbeeld zijn ze dat niet: de normen voor zwangerschap reizen mee naar beneden, naar verzorgingsproducten en vitamines. Hier reizen de normen voor iemands woonplaats mee naar de berichten. Zie *Omhoog in de voedselketen van data* in *Het idee*.",
            "right": "In het voorbeeld van Nissenbaum horen de normen voor zwangerschap mee naar beneden te reizen, naar de verzorgingsproducten en vitamines waaruit ze afgeleid werd. Zo reizen de normen voor iemands woonplaats mee naar de berichten waaruit die te raden is. Dat is de lezing van de course van het antwoord van Nissenbaum, toegepast op berichten.",
            "none": "Contextual integrity kent geen terrein waar alles mag: volgens Nissenbaum valt geen enkel deel van het leven buiten de normen voor stromen van informatie. En de berichten waren openbaar, iemands woonplaats niet. Zie *Contextual integrity* in *Het idee*.",
        },
    )
    choice_chain
    return (choice_chain,)


@app.cell(hide_code=True)
def text_exercise_rule():
    mo.md(r"""
    ### Oefening 5. De regel kennen

    In ConfAIde vertelde X aan Y een geheim, en alleen aan Y. Als Y de kans krijgt het aan Z te vertellen, krijgt ChatGPT twee vragen: rechtstreeks, of Y het geheim om die reden aan Z mag vertellen, en wat Y tegen Z zegt.
    """)
    return


@app.cell(hide_code=True)
def choice_rule_cell():
    choice_rule = L.choice(
        "rule",
        question="In minstens één van 10 runs beantwoordde ChatGPT de eerste vraag fout in 4% van de scenario's, en lekte het geheim in het antwoord aan Z in 93% ervan. Wat laat dat zien?",
        options={
            "unknown": "ChatGPT kent de regel niet",
            "right": "ChatGPT kan de regel noemen, maar niet toepassen",
            "broken": "De test klopt niet, want ze kunnen niet allebei waar zijn",
        },
        correct="right",
        why={
            "unknown": "Rechtstreeks gevraagd gaf ChatGPT bijna altijd het goede antwoord: Y mag het niet vertellen. Het ging mis toen ChatGPT dat moest toepassen in een eigen antwoord. Zie *Doorverteld* in *Het idee*.",
            "right": "In 96% van de scenario's beantwoordde ChatGPT de ja-neevraag in 10 runs nooit fout; en in 93% lekte het minstens één keer toen het het antwoord van Y schreef. Toen de auteurs het stap voor stap redeneren bij lekkende antwoorden lazen, noemde ChatGPT in 51,2% ervan privacy, en vertelde het geheim toch.",
            "broken": "Allebei kan: de twee vragen gaan over iets anders. Zeggen dat Y het niet mag vertellen is één taak, een antwoord aan Z schrijven zonder het geheim een andere. ChatGPT doet de eerste goed, en de tweede slecht. Zie *Doorverteld* in *Het idee*.",
        },
    )
    choice_rule
    return (choice_rule,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Wat is privacy?](/course/open/rai-what-is-privacy?lang=nl) zag je drie opvattingen van privacy, en aggregatie, die stukjes over één persoon samenvoegt. Hier kreeg de derde opvatting vijf parameters, en kreeg aggregatie een nieuwe vorm: een model dat van een paar berichten iemands woonplaats maakt, maakt een nieuwe stroom, van informatie die niemand deelde. In de volgende les, *Pseudoniem is niet anoniem*, kijk je wat er van een persoon over is in data zonder namen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_chain, choice_rule, result_allowed, result_differs, result_hits):
    L.wrap_up(result_differs, result_allowed, result_hits, choice_chain, choice_rule)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
