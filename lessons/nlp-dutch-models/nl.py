# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-dutch-models. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-dutch-models", "nl")
    C = L.checks   # de zinnen van de course, de tokenizers van GPT-2 en RobBERT, de tellingen van NTREX-128 en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Talen met weinig data](/course/open/nlp-low-resource?lang=nl) en [BERT: het ontbrekende woord invullen](/course/open/nlp-bert?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_premium():
    quick_premium = L.choice(
        "quick-premium",
        question="Op de eerste 100 zinnen van NTREX-128 maakt de tokenizer van GPT-2 5.414 tokens van de Nederlandse zinnen en 2.741 van de Engelse. Wat is de premium van het Nederlands voor GPT-2?",
        options={
            "upside": "ongeveer 0,51, het Engels gedeeld door het Nederlands",
            "right": "ongeveer 1,98: het Nederlands kost bijna twee keer zoveel tokens",
            "difference": "2.673, de extra tokens van het Nederlands",
            "both": "8.155, want een premium telt de tokens van beide talen bij elkaar op",
        },
        correct="right",
        why={
            "right": "De premium is het aantal tokens van een taal gedeeld door het aantal tokens van dezelfde tekst in het Engels: 5.414 / 2.741 = 1,98. Deze les gaat over modellen voor het Nederlands, en over wat de tokenizers van deze modellen van het Nederlands maken.",
            "upside": "Dat deelt andersom: het Engels hoort onder de streep. Welk getal komt erboven? Zie [Talen met weinig data](/course/open/nlp-low-resource?lang=nl).",
            "difference": "De premium deelt, en trekt niet af: de maat zegt hoeveel keer zoveel tokens. Wat deel je door wat? Zie [Talen met weinig data](/course/open/nlp-low-resource?lang=nl).",
            "both": "De premium vergelijkt de twee talen, dus de maat kan niet het totaal van beide zijn. Wat deel je door wat? Zie [Talen met weinig data](/course/open/nlp-low-resource?lang=nl).",
        },
    )
    quick_premium
    return (quick_premium,)


@app.cell(hide_code=True)
def quick_check_mask():
    quick_mask = L.choice(
        "quick-mask",
        question="BERT krijgt de zin *De trein naar Utrecht vertrekt over tien [MASK].* Wat doet BERT met de [MASK]?",
        options={
            "next": "het schrijft de woorden na de zin, één token tegelijk",
            "skip": "het laat de [MASK] weg",
            "right": "het raadt het stukje op de plek van de [MASK] uit de woorden aan beide kanten",
            "lookup": "het zoekt het meest voorkomende woord van het Nederlands op en zet dit woord op de plek",
        },
        correct="right",
        why={
            "right": "BERT leerde met masked language modeling: in een encoder kijkt elk token naar alle tokens, dus BERT raadt een verborgen stukje vanuit beide kanten. BERTje en RobBERT, in deze les, zijn BERT's die op dezelfde manier leerden van Nederlandse tekst.",
            "next": "Eén token tegelijk schrijven doet een decoder. Wat doet een encoder op een verborgen plek? Zie [BERT: het ontbrekende woord invullen](/course/open/nlp-bert?lang=nl).",
            "skip": "De [MASK] is precies de plek waarop BERT leerde werken. Wat doet BERT daar? Zie [BERT: het ontbrekende woord invullen](/course/open/nlp-bert?lang=nl).",
            "lookup": "BERT gebruikt de woorden rond de [MASK], geen lijst met veelvoorkomende woorden. Waaruit raadt BERT? Zie [BERT: het ontbrekende woord invullen](/course/open/nlp-bert?lang=nl).",
        },
    )
    quick_mask
    return (quick_mask,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Het Nederlands zit in groep 4 van Joshi en anderen: veel tekst, weinig gelabelde data. De modellen die de meeste mensen gebruiken leerden van tekst die vooral Engels was, en in [Talen met weinig data](/course/open/nlp-low-resource?lang=nl) zag je wat de tokenizers van deze modellen van het Nederlands maken: GPT-2 heeft voor hetzelfde nieuws bijna twee keer zoveel tokens nodig als in het Engels. Wil een Nederlandse organisatie een model, dan wil de organisatie vaak meer dan een model dat ook Nederlands begrijpt: een model dat goed Nederlands schrijft en Nederland kent, Nederlands in weinig tokens knipt, misschien op de eigen computers draait, en alleen leerde van tekst die het mocht gebruiken. Sinds 2019 heeft het Nederlands eigen modellen, gemaakt op drie manieren, en de manier van maken is terug te zien in wat een model kan. Deze les gaat over deze modellen: waarvan elk model leerde, en wat je daardoor van het model mag verwachten.

    **Een voorbeeld uit het dagelijks leven.** Een reisgids voor heel Europa geeft Amsterdam twee pagina's; een reisgids voor Nederland geeft Amsterdam er vijftig, en kent ook de veerboot naar Texel. Is er geen gids voor Nederland, dan is een Europese gids met een extra hoofdstuk over Nederland het beste alternatief.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_fietje = L.guess(
        "guess",
        question="Fietje is Phi-2, een model van 2,7 miljard parameters dat vooral op het Engels gericht is, verder getraind op 28 miljard tokens Nederlandse tekst. De tokenizer van RobBERT, gemaakt voor het Nederlands, heeft op de 1.997 Nederlandse zinnen van NTREX-128 1,31 tokens per woord nodig. Hoeveel tokens per Nederlands woord heeft de tokenizer van Fietje nodig?",
        options={
            "robbert": "ongeveer 1,3, net als RobBERT: van de 28 miljard Nederlandse tokens leerde de tokenizer Nederlandse stukjes",
            "between": "ongeveer 1,7, halverwege RobBERT en een Engelse tokenizer",
            "right": "ongeveer 2,2, evenveel als GPT-2",
            "more": "ongeveer 3,5: een tweede taal leren laat het model beide talen in meer stukjes knippen",
        },
        reveal="**2,22**, tot op het laatste cijfer evenveel als GPT-2: 104.053 tokens voor de 46.813 Nederlandse woorden. Verder trainen verandert de weights van een model, niet de tokenizer: Fietje hield de tokenizer van Phi-2, en Phi-2 gebruikt de tokenizer van GPT-2, met 38 extra tokens voor reeksen spaties en tabs in code. GEITje hield op dezelfde manier de tokenizer van Mistral: 2,01 tokens per Nederlands woord.",
    )
    guess_fietje
    return (guess_fietje,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Drie wegen naar een model voor het Nederlands.** De modellen voor het Nederlands ontstonden tot nu toe op drie manieren. *Vanaf het begin op het Nederlands*: een nieuw model leert alleen van Nederlandse tekst, met een tokenizer uit Nederlandse tekst; BERTje en RobBERT zijn encoders van deze soort. *Verder getraind*: een model leerde eerst van vooral Engelse tekst, en leert daarna verder van Nederlandse tekst, met de tokenizer die het al had; GEITje en Fietje zijn decoders van deze soort. *Vanaf het begin op veel talen*: een nieuw model leert tegelijk van Nederlands, Engels en andere talen, met een tokenizer voor al deze talen; EuroLLM en GPT-NL zijn decoders van deze soort.

    **BERTje.** Wietse de Vries en vijf anderen aan de Rijksuniversiteit Groningen trainden een BERT zo groot als BERT-base, ongeveer 109 miljoen parameters, op Nederlandse tekst: romans, het nieuwscorpus TwNC, SoNaR-500 uit [Tools en corpora voor het Nederlands](/course/open/nlp-dutch-tools?lang=nl), vier Nederlandse nieuwssites en Wikipedia, samen 12 GB tekst, ongeveer 2,4 miljard tokens. De vocabulary van 30.000 stukjes kwam uit deze tekst, en is cased: hoofdletters blijven staan. Gefinetuned op de namen in Nederlands nieuws van CoNLL-2002 haalde BERTje een span-F1 van 0,883, en mBERT, dat het Nederlands alleen van Wikipedia leerde, 0,807. Volgens de makers, in 2019, doet BERTje het op taken per woord steeds beter dan multilingual BERT.

    **RobBERT.** Pieter Delobelle, Thomas Winters en Bettina Berendt van de KU Leuven namen RoBERTa, de BERT die langer trainde op meer tekst, en trainden het model vanaf het begin op het Nederlandse deel van OSCAR, webtekst uit Common Crawl, een vrij beschikbare verzameling van miljarden webpagina's: 39 GB, 6,6 miljard woorden op 126 miljoen regels. RobBERT heeft 117 miljoen parameters. Er kwamen twee versies: v1 hield de Engelse tokenizer van RoBERTa, en v2 kreeg een nieuwe tokenizer van 40.000 tokens, gemaakt uit de Nederlandse tekst zoals de tokenizer van GPT-2 uit Engelse tekst gemaakt is. Bij *die* en *dat* zag je het verschil. Gevraagd welk van de twee op de plek van een [MASK] hoort, zonder enige fine-tuning, had v2 98,75% goed, v1 98,03%, BERTje 94,94% en mBERT 90,21%. En na fine-tuning op dezelfde taak, met de hele dataset, halveerde de Nederlandse tokenizer volgens de makers het aantal fouten. RobBERT-2022 kreeg daarna 2.774 tokens voor nieuwere woorden, zoals woorden over corona, en RobBERT-2023 kreeg een helemaal nieuwe tokenizer van 50.000 tokens.

    **GEITje: verder getraind, en offline gehaald.** In 2023 nam Edwin Rijgersberg Mistral 7B, een open model van 7 miljard parameters, en trainde het model verder op het Nederlands: op 10 miljard tokens Nederlandse tekst uit het Nederlandse Gigacorpus en het webcorpus MADLAD-400. Daarna leerde GEITje-chat van chatgesprekken, vertaald naar het Nederlands. GEITje hield de tokenizer van Mistral, van 32.000 tokens, en was open, onder de licentie Apache 2.0. In januari 2025 was GEITje weg. Stichting BREIN, een Nederlandse stichting die namens rechthebbenden optreedt tegen illegale kopieën, zei dat GEITje had geleerd van tienduizenden Nederlandse boeken uit een illegale bron, Library Genesis, en van tekst die zonder toestemming van nieuwssites was gekopieerd. De README begint nu met: "Op dringend verzoek van Stichting BREIN is GEITje vanaf heden niet meer beschikbaar." De herkomst van de trainingstekst bepaalt of je een model überhaupt mag gebruiken.

    **Fietje: klein, en verder getraind.** In 2024 deed Bram Vanroy hetzelfde met een klein model: Phi-2 van Microsoft, met 2,7 miljard parameters, getraind op 1,4 biljoen tokens, vooral Engels. Vanroy trainde het model verder op 28 miljard tokens Nederlands: de Nederlandse Wikipedia en Nederlandse webtekst uit CulturaX, gefilterd op kwaliteit. Fietje bestaat in drie versies, onder de licentie MIT: een versie die alleen een tekst aanvult, een versie die instructies volgt, en een versie voor gesprekken. Fietje hield de tokenizer van Phi-2. Volgens Vanroy doet deze tokenizer het wat slechter in fertility dan andere modellen: gemiddeld 2,05 subword-tokens per Nederlands woord, een teken dat de tokenizer voor het Engels bedoeld was. Vanroy mat dat op andere Nederlandse tekst; op NTREX-128 is het 2,22, zoals het plaatje hieronder laat zien. In de tests van Vanroy versloeg Fietje Chat, de versie voor gesprekken, op twee van de vijf taken GEITje Ultra, een versie van GEITje voor gesprekken en meer dan twee keer zo groot. En nieuwere modellen voor veel talen verslaan Fietje en GEITje allebei: volgens Vanroy doen recente kleine modellen het beter dan oudere, grotere modellen met fine-tuning voor het Nederlands.

    **EuroLLM: vanaf het begin op 35 talen.** EuroLLM komt van Unbabel, het Instituto Superior Técnico in Lissabon, de Universiteit van Edinburgh en anderen. De modellen leerden vanaf het begin van de 24 officiële talen van de Europese Unie, waaronder het Nederlands, en van 11 andere talen, zoals Arabisch, Chinees en Oekraïens. De modellen van 1,7, 9 en 22 miljard parameters leerden elk van ongeveer 4 biljoen tokens. In de hoofdfase was de helft Engels, 5% code en wiskunde, en de andere 45% ging naar de andere talen. De tokenizer is voor al deze talen gemaakt: volgens de makers is een vocabulary van 128.000 stukjes het beste compromis, vier keer de 32.000 van Mistral. De licentie is Apache 2.0.

    **GPT-NL: vanaf het begin op rechtmatige data.** GPT-NL is een model van TNO, SURF en het Nederlands Forensisch Instituut, gefinancierd met 13,5 miljoen euro van de Nederlandse overheid. Het model leerde vanaf het begin, van juni tot december 2025, alleen van data die de makers rechtmatig verkregen: tekst met een licentie van de eigenaren, teksten uit Common Crawl onder open licenties, synthetische tekst en code. De site geeft de mix: ongeveer 10% Nederlands, 73% Engels en 17% andere talen, en tijdens de training zag het model de Nederlandse teksten vaker dan de andere. Een gelijke verdeling van Nederlands en Engels bleek niet haalbaar: met alleen rechtmatige data putten de makers uit minder bronnen dan andere ontwikkelaars. GPT-NL heeft 26 miljard parameters en een tokenizer van 128.000 tokens, voor ongeveer 40% Nederlands. De weights zijn niet open: onderzoekers krijgen de weights voor een kleine vergoeding, alle anderen met een betaalde licentie, en in november 2026 gebruiken alleen een paar launching customers het model, zoals de KB, de nationale bibliotheek. En OpenEuroLLM, 20 Europese partners met 37,4 miljoen euro, deels van de EU, bouwt open modellen voor alle officiële talen van de EU.

    **Encoder of decoder.** BERTje en RobBERT zijn encoders, net als BERT: deze modellen lezen een hele tekst en geven labels, na fine-tuning op gelabelde voorbeelden, zoals in [Een encoder finetunen voor een taak](/course/open/nlp-fine-tune-bert?lang=nl). Met ongeveer 110 miljoen parameters zijn BERTje en RobBERT klein en snel. GEITje, Fietje, EuroLLM en GPT-NL zijn decoders: deze modellen schrijven tekst, met miljarden parameters.

    **Wat de tokenizer kost.** Het plaatje vergelijkt de tokenizers van zeven modellen op alle 1.997 zinnen van NTREX-128, in het Nederlands en in het Engels, in tokens per woord: de tokenizer fertility uit [Talen met weinig data](/course/open/nlp-low-resource?lang=nl). De tokenizer van RobBERT heeft 1,31 tokens per Nederlands woord nodig, de tokenizer van BERTje 1,32, en beide hebben meer nodig voor het Engels: 1,74 en 2,11. De tokenizer van EuroLLM, gemaakt voor 35 talen, heeft 1,47 nodig voor het Nederlands en 1,35 voor het Engels; de tokenizer van Qwen3 1,83 en 1,26. De tokenizer van Mistral heeft 2,01 nodig voor het Nederlands, en de tokenizer van Fietje knipt beide talen precies zoals de tokenizer van GPT-2. Verder trainen leert een model Nederlands, maar verandert niets aan de kosten van de tokenizer.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.md(r"""
    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **BERTje** | een Nederlandse BERT van de Rijksuniversiteit Groningen (2019), vanaf het begin getraind op 12 GB Nederlandse boeken, nieuws en Wikipedia |
    | **RobBERT** | een Nederlandse RoBERTa van de KU Leuven (2020), getraind op Nederlandse webtekst met een tokenizer voor het Nederlands; RobBERT-2023 is de nieuwste versie |
    | **GEITje** | Mistral 7B, verder getraind op 10 miljard tokens Nederlands (2023); in 2025 offline gehaald om de herkomst van een deel van de tekst |
    | **Fietje** | Phi-2, 2,7 miljard parameters, verder getraind op 28 miljard tokens Nederlands (2024), met de tokenizer van Phi-2 |
    | **GPT-NL** | een Nederlands model van TNO, SURF en het NFI, vanaf het begin getraind op rechtmatig verkregen data, voor ongeveer 10% Nederlands; de weights zijn niet open |
    | **EuroLLM** | open modellen, vanaf het begin getraind op 35 talen, waaronder de 24 officiële talen van de EU, met een tokenizer van 128.000 tokens |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo kies je een model voor een taak in het Nederlands:

    1. Bepaal wat de taak teruggeeft. Een label per tekst of per woord: dan volstaat een encoder zoals RobBERT. Nieuwe tekst: dan heb je een decoder nodig.
    2. Meet wat de tokenizer van het model van je Nederlandse tekst maakt: tokens per woord, en hoeveel van de tekst in de context past, de tokens die het model tegelijk leest.
    3. Lees waarvan het model leerde: hoeveel Nederlands, uit welke bronnen, en onder welke licentie.
    4. Probeer het model op voorbeelden van je eigen taak, naast een nieuwer model dat veel talen leerde.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `C.OWN` bevat de drie zinnen van de course in het Nederlands, `C.OWN["nl"]`, en dezelfde zinnen in het Engels, `C.OWN["en"]`. `C.ENC` is de tokenizer van GPT-2, ook de tokenizer van Fietje, en `C.ROBBERT` de tokenizer van RobBERT v2. Beide hebben `encode`, dat de ids van de tokens van een tekst geeft, en `decode`, dat van een lijst met ids weer tekst maakt. `C.ROBBERT.pieces(text)` geeft de tokens in de vorm van de vocabulary, met Ġ voor een spatie. `C.KEYS` noemt de zeven tokenizers van het plaatje, en `C.counts()` geeft voor elke tokenizer de tellingen van de course over alle 1.997 zinnen van NTREX-128: de woorden en de tokens, in het Nederlands en in het Engels. `C.per_word(key, code)` rekent met deze tellingen het aantal tokens per woord uit, met `"nld"` voor het Nederlands en `"eng"` voor het Engels:
    """)
    return


@app.cell
def example_cell():
    _sentence = C.OWN["nl"][0]
    _gpt2 = C.ENC.encode(_sentence)
    _robbert = C.ROBBERT.encode(_sentence)
    print(len(_gpt2), _gpt2[:6], C.ENC.decode(_gpt2))
    print(len(_robbert), _robbert[:6], C.ROBBERT.decode(_robbert))
    print(C.ROBBERT.pieces(_sentence))
    for _key in C.KEYS:
        print(_key, C.counts()[_key]["nld_tokens"], round(C.per_word(_key, "nld"), 2), round(C.per_word(_key, "eng"), 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    GPT-2 maakt 29 tokens van de Nederlandse zin over het fietspad, RobBERT 13: één per woord, en één voor de punt. De ids zijn niet gelijk, `[5005, 16840, 68, ...]` tegenover `[62, 512, 2733, ...]`, want elke tokenizer nummert de eigen tokens, en `decode` geeft in beide gevallen de zin terug. De stukjes van RobBERT laten zien waarom RobBERT zo weinig tokens nodig heeft: *Ġgemeente*, *Ġfietspad* en *Ġkanaal* zijn elk één token: het hele woord, met een spatie vooraan.

    Dan NTREX-128. GPT-2 en Fietje maken 104.053 tokens van de 46.813 Nederlandse woorden, 2,22 per woord, en 1,24 per Engels woord. RobBERT maakt er 61.429, 1,31 per woord, en heeft 1,74 per Engels woord nodig: RobBERT leerde de stukjes uit het Nederlands, dus nu is het Engels de taal die meer kost.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pieces():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. pieces: hoe knipt een tokenizer een woord?

        Schrijf `pieces`, met `word`, een woord, en `enc`, een tokenizer. De functie geeft de stukjes van het woord terug als een lijst met teksten, met een spatie vooraan, zoals het woord in een zin staat. In de startcode begint `out` als een lege lijst, en `for token in enc.encode(" " + word)` loopt langs de ids van de tokens. `out.append(piece)` voegt elk stukje toe aan de lijst. Vervang de `...` door de tekst van één token, `token`: `enc.decode` maakt van een lijst met ids tekst. Dezelfde loop schreef je in [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl) met één vaste tokenizer; hier is de tokenizer een argument, `enc`, zodat één functie met elke tokenizer knipt.

        De check roept `pieces("fietsenstalling", C.ENC)` aan, met de tokenizer van GPT-2, wat `[' fi', 'ets', 'en', 'st', 'alling']` moet geven. Daarna eigen gevallen: Nederlandse en Engelse woorden, met de tokenizer van GPT-2 en met de tokenizer van RobBERT, `C.ROBBERT`.
        """),
        L.task("Geef `piece` de tekst van het ene token `token`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pieces():
    box_pieces = L.code_box("pieces", lines=6)
    box_pieces
    return (box_pieces,)


@app.cell(hide_code=True)
def check_pieces(box_pieces):
    result_pieces = L.check("pieces", box_pieces.value)
    result_pieces
    return (result_pieces,)


@app.cell(hide_code=True)
def hints_pieces():
    L.hints(
        "pieces",
        [
            "`piece` is de tekst van één token. `enc.decode` maakt van een lijst met ids tekst, dus geef de methode een lijst met alleen `token`.",
            "`C.ENC.decode([15496, 0])` geeft `\"Hello!\"`, en `C.ENC.decode([0])` geeft `\"!\"`: een lijst met één id geeft de tekst van dat ene token.",
            "`        piece = enc.decode(___)`",
        ],
        why="`piece = enc.decode([token])`: `decode` wil een lijst, en `[token]` is een lijst met één id. Zonder de haken stopt `enc.decode(token)` met een `TypeError`.",
    )
    return


@app.cell(hide_code=True)
def text_after_pieces():
    mo.md(r"""
    In het plaatje knipt je functie vijf Nederlandse woorden en twee Engelse met beide tokenizers. GPT-2 knipt elk Nederlands woord in drie tot vijf stukjes, en de stukjes volgen meestal niet de delen van het woord: *ziekenhuis* wordt *z*, *ie*, *ken*, *hu* en *is*. De tokenizer van RobBERT houdt *gemeenteraad*, *ziekenhuis*, *broodjes* en *uitzendbureau* heel, en knipt *fietsenstalling* in *fietsen* en *stalling*. Bij het Engels is het andersom: GPT-2 houdt *library* en *Saturday* heel, en RobBERT knipt *Saturday* in *Sat*, *ur* en *day*. Een tokenizer leert de stukjes uit de tekst waaruit de tokenizer gemaakt is.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_words_in_context():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. words_in_context: hoeveel Nederlands past erin?

        Een model leest een vast aantal tokens tegelijk, de context: Fietje 2.048, net als Phi-2. Schrijf `words_in_context`, met `context`, een aantal tokens, `texts`, een lijst met zinnen, en `enc`, een tokenizer. De functie geeft terug hoeveel woorden van zulke tekst in `context` tokens passen. In de startcode beginnen `tokens` en `words` bij 0, en `for text in texts` loopt langs de zinnen: `tokens = tokens + len(enc.encode(text))` telt de tokens, en `words = words + len(text.split())` de woorden. Vervang de `...` na `return` door het aantal woorden dat past: `context` keer het deel van een woord dat één token gemiddeld bevat.

        De check roept `words_in_context(2048, ["De gemeente legt volgend jaar een nieuw fietspad aan langs het kanaal."], C.ENC)` aan, wat `847.448275862069` moet geven: 29 tokens voor 12 woorden. Daarna eigen gevallen: de zinnen van de course in het Nederlands en in het Engels, met beide tokenizers, en contexten van 512 en 8.192 tokens.
        """),
        L.task("Geef het aantal woorden terug dat in `context` tokens past."),
    ])
    return


@app.cell(hide_code=True)
def exercise_words_in_context():
    box_words_in_context = L.code_box("words_in_context", lines=8)
    box_words_in_context
    return (box_words_in_context,)


@app.cell(hide_code=True)
def check_words_in_context(box_words_in_context):
    result_words_in_context = L.check("words_in_context", box_words_in_context.value)
    result_words_in_context
    return (result_words_in_context,)


@app.cell(hide_code=True)
def hints_words_in_context():
    L.hints(
        "words_in_context",
        [
            "Eén token bevat gemiddeld de woorden van de teksten gedeeld door de tokens. Vermenigvuldig dat met `context`.",
            "Bevatten 17 tokens 8 woorden, dan bevat één token 8 / 17 van een woord, en bevatten 100 tokens 100 × 8 / 17, ongeveer 47 woorden.",
            "`    return context * ___`",
        ],
        why="`return context * words / tokens`: `words / tokens` is het deel van een woord dat één token bevat, het omgekeerde van de tokens per woord uit [Talen met weinig data](/course/open/nlp-low-resource?lang=nl). `context / tokens * words` geeft hetzelfde.",
    )
    return


@app.cell(hide_code=True)
def text_after_words_in_context():
    mo.md(r"""
    In het plaatje rekent je functie met de drie zinnen van de course: in 2.048 tokens passen met de tokenizer van GPT-2 970 woorden van het Nederlands en 1.787 van het Engels, en met de tokenizer van RobBERT 1.843 woorden Nederlands en 1.354 Engels. Daaronder staan de tellingen van de course over NTREX-128 voor alle zeven tokenizers: de 2.048 tokens van Fietje bevatten 921 Nederlandse woorden, en met de tokenizer van RobBERT zouden er 1.561 in passen. Knipt de tokenizer van een model het Nederlands in meer stukjes, dan leest het model minder Nederlands tegelijk, schrijft het model Nederlands trager, één token tegelijk, en kost elk Nederlands woord meer bij een dienst die per token rekent.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_share():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. share: van hoeveel Nederlands leerde een model?

        Schrijf `share`, met `mix`, de trainingstekst van een model als een dictionary van een taal naar de miljarden tokens van de taal, en `language`, een van de sleutels. De functie geeft het aandeel van deze taal terug: de tokens van de taal gedeeld door alle tokens. In de startcode begint `total` bij 0, en `for name, tokens in mix.items()` telt de tokens van elke taal erbij op. Vervang de `...` na `return` door het aandeel van `language`.

        De check roept `share({"nl": 29, "en": 208, "other": 48}, "nl")` aan, de tekst van GPT-NL, wat `0.10175438596491228` moet geven. Daarna eigen gevallen: andere talen, en de trainingstekst van andere modellen.
        """),
        L.task("Geef het aandeel van `language` in `mix` terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_share():
    box_share = L.code_box("share", lines=6)
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
            "De tokens van `language` staan in `mix`, onder de sleutel `language`. Deel deze tokens door `total`.",
            "`{\"nl\": 29, \"en\": 208}[\"en\"]` geeft `208`: vierkante haken na een dictionary geven de waarde van een sleutel.",
            "`    return ___ / total`",
        ],
        why="`return mix[language] / total`: de tokens van één taal gedeeld door de tokens van alle talen, een getal van 0 tot 1.",
    )
    return


@app.cell(hide_code=True)
def text_after_share():
    mo.md(r"""
    In het plaatje rekent je functie met de trainingstekst van vier modellen. BERTje leerde alleen van Nederlands: 100%. GPT-NL, een model voor het Nederlands, leerde van tekst met 10,2% Nederlands. De 28 miljard Nederlandse tokens van Fietje zijn 2,0% van alle tokens waarvan het model leerde, de 1,4 biljoen van Phi-2 meegeteld. OpenEuroLLM gaf het Nederlands in Prelude, het eerste testmodel, uit 2026, 47,5 miljard van de 7.815 miljard tokens van de eerste fase: 0,6%, meer Nederlandse tokens dan GEITje en Fietje samen, in een veel groter geheel. Voor GEITje is geen aandeel te geven: Mistral heeft niet bekendgemaakt van hoeveel tekst Mistral 7B leerde.

    ### Probeer het: je eigen tekst, zeven tokenizers

    De demo heeft de tokenizers van BERTje, Mistral 7B, Fietje, EuroLLM en Qwen3 nodig. Typ een tekst in het vak: het plaatje knipt de tekst met alle zeven tokenizers, de minste tokens eerst.

    Begin met de zin in het vak. Probeer dan een lange Nederlandse samenstelling, zoals *arbeidsongeschiktheidsverzekering*, los en in een zin, daarna een zin in het Engels, en *Eén café, twee cafés.*
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label=f"Download de vijf tokenizers, {C.download_size(C.FILES, L.lang)}")
    return (get_button,)


@app.cell(hide_code=True)
def tokenizers_cell(get_button):
    C.files_or_stop(get_button, {
        "ask": f"*De tokenizers van BERTje, Mistral 7B, Fietje, EuroLLM en Qwen3 staan nog niet op je computer. De knop downloadt de bestanden één keer van Hugging Face, {C.download_size(C.FILES, L.lang)}: de tokenizers van Mistral 7B, EuroLLM en Qwen3 onder de licentie Apache-2.0, de tokenizer van Fietje onder MIT, en de vocabulary van BERTje van de pagina van de makers. De rest van de les werkt zonder.*",
        "busy": "**Bezig met downloaden** {done} van {total} MB …",
        "no_net": "*Het downloaden lukte niet: Hugging Face was niet bereikbaar, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*Het downloaden lukte niet: een binnengekomen bestand is niet het vastgelegde bestand van de course, dus de course gooide het bestand weg. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*Het downloaden lukte niet: er is op je schijf geen plek voor de bestanden. Maak ruimte, en klik opnieuw.*",
        "failed": "*Het downloaden lukte niet ({why}). Klik opnieuw.*",
    })
    try_text = mo.ui.text_area(value=C.TRY_START, label="Je tekst", full_width=True, rows=3)
    try_text
    return (try_text,)


@app.cell(hide_code=True)
def try_look(try_text):
    mo.Html(C.try_picture(L.lang, try_text.value))
    return


@app.cell(hide_code=True)
def text_after_try():
    mo.md(r"""
    In de zin over de gemeenteraad hebben RobBERT en BERTje 16 tokens nodig voor 13 woorden, en beide houden *gemeenteraad*, *vanavond* en *binnenstad* heel. EuroLLM heeft er 21 nodig, Qwen3 26, Mistral, en dus ook GEITje, 32, en GPT-2 en Fietje elk 34, token voor token gelijk. Los getypt kost *arbeidsongeschiktheidsverzekering* GPT-2 12 tokens en RobBERT 6; in een zin, na een spatie, kost het woord RobBERT er 2, *arbeidsongeschiktheids* en *verzekering*. RobBERT leerde de stukjes uit doorlopende tekst, en daarin staat een woord bijna altijd na een spatie. In een Engelse zin draait de volgorde om: RobBERT en BERTje hebben de meeste tokens nodig. En in *Eén café, twee cafés.* knipt GPT-2 *Eén*, *twee* en *cafés* elk in tweeën, terwijl BERTje en RobBERT elk woord heel houden, ook *Eén* met de hoofdletter.

    ### Oefening 4. Encoder of decoder?

    Je hebt zes taken met Nederlandse tekst. Deel elke taak in: volstaat een encoder zoals RobBERT, of vraagt de taak om een decoder zoals Fietje?
    """)
    return


@app.cell(hide_code=True)
def exercise_fits():
    sort_fits = L.sort("fits")
    sort_fits
    return (sort_fits,)


@app.cell(hide_code=True)
def check_fits(sort_fits):
    result_fits = L.check("fits", sort_fits)
    result_fits
    return (result_fits,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** BERTje en RobBERT zijn BERT's, zoals in [BERT: het ontbrekende woord invullen](/course/open/nlp-bert?lang=nl), en je finetunet deze modellen voor een taak zoals in [Een encoder finetunen voor een taak](/course/open/nlp-fine-tune-bert?lang=nl). BERTje leerde deels van SoNaR, uit [Tools en corpora voor het Nederlands](/course/open/nlp-dutch-tools?lang=nl). Tokens per woord is de tokenizer fertility uit [Talen met weinig data](/course/open/nlp-low-resource?lang=nl): een tokenizer voor het Nederlands brengt de maat terug tot ongeveer 1,3, en verder trainen verandert niets aan de tokenizer en de kosten. Met deze les eindigen de module over meer talen en de basis van het domein Taal en zoeken.

    **Een voorbeeld uit het dagelijks leven.** Een Engels kookboek in Nederlandse vertaling is in het Nederlands geschreven, maar meet vaak nog in cups en ounces: de vertaler veranderde de woorden, niet de maten. Verder trainen op het Nederlands verandert wat een model weet, maar niet de tokenizer: de maat waarmee het model tekst knipt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_pieces, result_words_in_context, result_share, result_fits):
    L.wrap_up(result_pieces, result_words_in_context, result_share, result_fits)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
