# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-low-resource. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-low-resource", "nl")
    C = L.checks   # de zinnen van de course in zeven talen, de tellingen van NTREX-128, de tokenizers en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl) en [Zo'n 7.000 talen](/course/open/nlp-languages-of-the-world?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_tokens():
    quick_tokens = L.choice(
        "quick-tokens",
        question="GPT-2 knipt de Engelse zin *Tokenization is surprisingly important.* in 6 tokens, en dezelfde zin in het Nederlands, *Tokenisatie is verrassend belangrijk.*, in 15. Waar komt het verschil vandaan?",
        options={
            "longer": "de Nederlandse zin heeft meer woorden",
            "letters": "GPT-2 knipt elk woord van een andere taal dan het Engels in losse letters",
            "right": "GPT-2 maakte de lijst met tokens uit tekst die bijna helemaal Engels was",
            "random": "een tokenizer knipt op willekeurige plekken, dus het aantal verschilt per keer",
        },
        correct="right",
        why={
            "right": "De tokens van GPT-2 zijn de stukjes die in de trainingstekst vaak samen voorkwamen, en deze tekst was bijna helemaal Engels. Engelse woorden zijn dus vaak één token, en Nederlandse woorden vallen in stukjes: *verrassend* werd er vier. Deze les meet hoeveel meer andere talen kosten, en wat dat betekent voor talen met nog minder tekst.",
            "longer": "Beide zinnen hebben vier woorden. Waaruit leerde GPT-2 de tokens? Zie [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl).",
            "letters": "*verrassend* werd vier stukjes, geen tien letters, en *is* bleef heel. Waar kwamen de stukjes van GPT-2 vandaan? Zie [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl).",
            "random": "Een tokenizer knipt dezelfde tekst elke keer op dezelfde manier, en `decode` geeft de tekst precies terug. Waaruit leerde GPT-2 de tokens? Zie [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl).",
        },
    )
    quick_tokens
    return (quick_tokens,)


@app.cell(hide_code=True)
def quick_check_classes():
    quick_classes = L.choice(
        "quick-classes",
        question="Pratik Joshi en vier anderen deelden 2.485 talen in zes groepen in, naar de data en hulpmiddelen die er voor elke taal zijn. Hoeveel van deze talen zitten in de laagste groep, de *Left-Behinds*?",
        options={
            "half": "ongeveer de helft",
            "seven": "7, evenveel als in de hoogste groep",
            "right": "2.191, bijna negen op de tien",
            "few": "een paar honderd, omdat de meeste talen wel wat tekst op het web hebben",
        },
        correct="right",
        why={
            "right": "2.191 van de 2.485 talen zitten in groep 0, met bijna geen digitale tekst, gelabeld of niet, en 7 in groep 5, de *Winners*. Het Nederlands zit in groep 4. Deze les gaat over wat een tokenizer en een model doen met de talen lager in de lijst, en hoe zo'n taal toch een model krijgt dat werkt.",
            "half": "Veel meer dan de helft: de groepen zijn heel ongelijk. Hoeveel talen zaten er in de laagste groep? Zie [Zo'n 7.000 talen](/course/open/nlp-languages-of-the-world?lang=nl).",
            "seven": "7 is de hoogste groep, de *Winners*, met het Engels, het Spaans, het Duits, het Japans en het Frans. Hoeveel zitten er onderaan? Zie [Zo'n 7.000 talen](/course/open/nlp-languages-of-the-world?lang=nl).",
            "few": "De meeste talen hebben bijna geen tekst op het web. Hoeveel van de 2.485 zitten in de laagste groep? Zie [Zo'n 7.000 talen](/course/open/nlp-languages-of-the-world?lang=nl).",
        },
    )
    quick_classes
    return (quick_classes,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Joshi en anderen zetten het Engels, het Spaans, het Duits, het Japans en het Frans in de hoogste groep: de rijkste talen, en elke doorbraak in NLP komt deze talen ten goede. Het Nederlands zit een groep lager, in groep 4, de *Underdogs*: talen met bijna evenveel ongelabelde tekst als de hoogste groep, maar met minder gelabelde data. Het Swahili, het Zoeloe en het Yoruba, elk met miljoenen sprekers, zitten in groep 2.

    Een model leert van de tekst die het krijgt. De tokenizer leert de stukjes ook uit deze tekst, en als de tekst vooral Engels is, kost elke andere taal meer tokens voor hetzelfde bericht. Rekent een dienst per token, dan betaal je voor zo'n taal meer, past er minder tekst in de context van het model, en schrijft het model het antwoord langzamer, één token tegelijk. Orevaoghene Ahia en zes anderen maten dit in 2023 voor de API van OpenAI: volgens de onderzoekers betalen de sprekers van veel van de ondersteunde talen te veel, en krijgen deze sprekers ook nog slechtere resultaten. Deze les meet wat een taal een tokenizer kost, en kijkt wat een taal met weinig data helpt aan een model dat werkt.

    **Een voorbeeld uit het dagelijks leven.** Een sms op een telefoon bevat 160 tekens uit het Latijnse alfabet. Zet er één Russische letter in, en het bericht bevat er nog 70: de telefoon slaat elk teken dan op in meer ruimte. Hetzelfde nieuws kost in het Russisch meer berichten dan in het Engels.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_tamil = L.guess(
        "guess",
        question="NTREX-128 is een set van 1.997 zinnen nieuws, door professionele vertalers uit het Engels vertaald in 128 talen. De tokenizer van GPT-2 maakt 2.741 tokens van de eerste 100 Engelse zinnen. Hoeveel tokens maakt de tokenizer van dezelfde 100 zinnen in het Tamil, een taal uit het zuiden van India, Sri Lanka en Singapore?",
        options={
            "same": "ongeveer 2.700: hetzelfde nieuws kost in elke taal evenveel",
            "dutch": "ongeveer 5.400, twee keer zoveel, net als het Nederlands",
            "five": "ongeveer 14.000, vijf keer zoveel",
            "right": "ongeveer 39.000, veertien keer zoveel",
        },
        reveal="**39.232**, 14,3 keer zoveel als de Engelse zinnen. Een letter van het Tamil is in UTF-8 drie bytes, en GPT-2 leerde bijna geen stukjes van het Tamil: de tokenizer maakt van elke byte 0,99 tokens. Qwen3 is een reeks modellen uit 2025, getraind op 119 talen en dialecten, en de tokenizer van Qwen3 maakt 15.772 tokens van de Tamilzinnen: 5,7 keer de 2.770 tokens van de Engelse zinnen. Het Nederlands kost GPT-2 5.414 tokens, 2,0 keer zoveel, en Qwen3 4.419, 1,6 keer.",
    )
    guess_tamil
    return (guess_tamil,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Low-resource language.** Een taal met weinig digitale tekst, en nog minder gelabelde voorbeelden voor een taak, zoals de namen in een tekst. Een scherpe grens is er niet. Joshi en anderen telden voor elke taal de gelabelde datasets in twee catalogi en de pagina's op Wikipedia, en deelden de talen op basis van deze tellingen in zes groepen in. Een taal kan veel sprekers hebben en toch weinig data: groep 0 bevat 88% van de ingedeelde talen en 15% van alle sprekers.

    **Een tokenizer leert van de eigen tekst.** De tokenizer van GPT-2 begint bij de 256 mogelijke waarden van een byte, en kan dus elke tekst schrijven. De tokenizer maakt tokens van stukjes die in de trainingstekst vaak samen voorkwamen, zoals je zag in [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl). Deze tekst was bijna helemaal Engels. Engelse woorden zijn dus vaak één token, Nederlandse woorden twee of drie, en de letters van een taal blijven los, of zelfs in bytes, als de tokenizer de taal zelden zag. In UTF-8 is een letter van het Latijnse alfabet zonder accent één byte, een Griekse of Russische letter twee bytes, en een letter van het Tamil, het Hindi of het Thai, of een Chinees teken, drie.

    **De premium.** Aleksandar Petrov, Emanuele La Malfa, Philip Torr en Adel Bibi vergeleken talen op dezelfde tekst, vertaald: de **premium** van een taal is het aantal tokens gedeeld door het aantal tokens van dezelfde zinnen in het Engels. Een premium van 1 is **pariteit**. De vier onderzoekers pleiten voor pariteit tussen talen: een tokenizer hoort voor dezelfde inhoud in elke taal ongeveer even lange reeksen tokens te maken. Op 2.000 zinnen uit Wikipedia, vertaald in 200 talen, vonden de onderzoekers voor GPT-2 een premium van 1,97 voor het Nederlands. Andere talen kosten veel meer: met de tokenizer van ChatGPT en GPT-4 kost het Shan, een taal uit Myanmar, 15 keer zoveel tokens als het Engels. Dezelfde tekst kan in een andere taal dus tot 15 keer zoveel tokens kosten.

    **Tokens per byte.** Het aantal tokens gedeeld door het aantal bytes van de tekst in UTF-8. Een token van GPT-2 bevat minstens één byte, dus deze maat is nooit meer dan 1, en dicht bij 1 betekent dat de tokenizer bijna geen stukjes van de taal leerde. Van het Engels maakt GPT-2 0,22 tokens per byte: een token bevat gemiddeld 4,6 bytes. Anders dan de premium heeft tokens per byte geen vertaling nodig: één tekst is genoeg.

    **Tokenizer fertility en continued words.** Phillip Rust en vier anderen meten een tokenizer op de woorden van een taal. De **tokenizer fertility** is het gemiddelde aantal stukjes per woord; een fertility van 1 is het minimum en betekent dat elk woord van de tekst als token in de lijst van de tokenizer staat. Dat is je `tokens_per_word` uit *Tokens*. De tweede maat telt de **continued words**, de woorden die de tokenizer in meer dan één stukje knipt: de fertility zegt hoe fijn de tokenizer knipt, en de continued words zeggen hoe vaak de tokenizer een woord knipt. Beide maten hebben woorden nodig, en `split` vindt alleen woorden waar een taal spaties tussen de woorden zet. Het Chinees, het Japans, het Thai, het Laotiaans, het Khmer en het Tibetaans schrijven de woorden zonder spaties. Ahia en collega's tellen om deze reden tokens per zin: voor veel van de onderzochte talen bestaat geen programma om de woorden te vinden, en zonder woorden geen fertility.

    **Wat de tokenizer met het model doet.** Meer tokens is meer dan een hogere rekening. Rust en collega's vergeleken modellen voor één taal met een meertalig model. Volgens de onderzoekers telt de hoeveelheid tekst voor een model mee, maar telt een tokenizer voor de taal zelf even zwaar voor hoe goed het model een taak doet. En als de onderzoekers de meertalige tokenizer van een meertalig model vervingen door een tokenizer voor één taal, werd het model beter in bijna elke taak en elke taal.

    **Dezelfde 100 zinnen in 48 talen.** Het plaatje toont de premium van 47 talen tegenover het Engels, op de eerste 100 zinnen van NTREX-128, voor twee tokenizers: de tokenizer van GPT-2, uit 2019, met 50.257 tokens, en de tokenizer van Qwen3, met 151.669. Qwen3 gebruikt de tokenizer die Qwen in 2023 voor de eerste eigen modellen maakte, en de modellen van Qwen3 uit 2025 leerden van tekst in 119 talen en dialecten. De premium van GPT-2 voor het Nederlands is 1,98, dicht bij de 1,97 van Petrov en collega's op andere zinnen. Het Spaans, het Frans, het Nederlands en het Duits kosten GPT-2 ongeveer twee keer zoveel tokens als het Engels, en het Samoaans en het Maori ook, uit groep 1, in het Latijnse alfabet. Het Russisch kost 5,55, het Arabisch 4,48, allebei uit de hoogste groepen: de premium volgt meer het schrift van een taal dan de groep. Qwen3 verlaagt elke premium: het Chinees van 3,21 naar 1,11, bijna pariteit, en het Russisch van 5,55 naar 1,88. En het Telugu, het Tibetaans, het Tamil, het Khmer, het Laotiaans en het Armeens kosten Qwen3 nog altijd meer dan vijf keer zoveel tokens als het Engels, en het Bengaals vijf keer zoveel.

    **Cross-lingual transfer.** Een taal met weinig gelabelde voorbeelden kan voorbeelden lenen. Een **meertalig model** leerde van tekst in veel talen tegelijk: Multilingual BERT, mBERT, leerde van Wikipedia in 104 talen, met één lijst stukjes voor alle talen samen. Telmo Pires, Eva Schlinger en Dan Garrette deden fine-tuning met de trainingsdata van een taak in één taal, en testten de taak in een andere taal. Dat is **cross-lingual transfer**: zero-shot, als het model geen enkel gelabeld voorbeeld ziet van de taal waarop het getest wordt. Na fine-tuning op de namen in het Engelse nieuws van CoNLL 2003 vond mBERT de namen in Nederlands nieuws met een span F1 van 0,774; na fine-tuning op Nederlands nieuws 0,899. Transfer werkt het best tussen talen die op elkaar lijken. Bij woordsoorten haalde mBERT gemiddeld een accuracy van 81,55% van een taal met de volgorde onderwerp, werkwoord, lijdend voorwerp naar een andere taal met deze volgorde, en 66,52% naar een taal met het werkwoord achteraan.

    **Een paar eigen voorbeelden.** Anne Lauscher en drie anderen voegden na de fine-tuning in het Engels een paar gelabelde zinnen uit de taal zelf toe. Bij sommige talen werd dependency parsing zo tot 40 UAS-punten beter, met niet meer dan 10 zinnen in de doeltaal. UAS is het deel van de woorden dat in een dependency tree het goede hoofdwoord krijgt, in punten van 100. Gemiddeld over de talen van het onderzoek tilden 500 zinnen uit de taal zelf de UAS van mBERT van 52,96 naar 80,47, en de accuracy bij woordsoorten van 67,2 naar 91,16. Bij taken over de betekenis van een hele zin, zoals de vraag of de ene zin uit de andere volgt, was de winst klein: daar maakten 100 voorbeelden de score zelfs 1,00 punt lager. En een eigen model op de weinige tekst van een taal helpt niet: volgens Shijie Wu en Mark Dredze doet mBERT het bij talen met veel data even goed als andere modellen of beter, maar bij talen met weinig data veel slechter, en een eentalig BERT-model voor zo'n taal doet het nog slechter. Waar gelabelde data ontbreekt, maken mensen gelabelde data. In MasakhaNER, een project van 61 onderzoekers, markeerden mensen de namen in teksten in tien Afrikaanse talen, waaronder het Swahili, het Wolof en het Yoruba: volgens de makers de eerste grote, openbare dataset van hoge kwaliteit voor named entity recognition in deze tien talen.
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
    | **low-resource language** | een taal met weinig digitale tekst, en weinig gelabelde voorbeelden voor taken |
    | **premium** | de tokens van een tekst in een taal gedeeld door de tokens van dezelfde tekst in het Engels |
    | **pariteit** | een premium van ongeveer 1: hetzelfde bericht kost in beide talen evenveel tokens |
    | **tokens per byte** | de tokens van een tekst gedeeld door de bytes in UTF-8; dicht bij 1 leerde de tokenizer bijna niets van de taal |
    | **tokenizer fertility** | het gemiddelde aantal tokens per woord van een taal; kortweg *fertility* |
    | **continued words** | het deel van de woorden dat de tokenizer in meer dan één token knipt |
    | **meertalig model** | een model dat van tekst in veel talen tegelijk leerde, zoals mBERT |
    | **cross-lingual transfer** | een meertalig model finetunen in één taal en het model gebruiken in een andere taal |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo meet je wat een taal een tokenizer kost:

    1. Neem zinnen die in de taal en in het Engels hetzelfde zeggen: vertalingen, zin voor zin.
    2. Tel de tokens van elke zin en tel alles bij elkaar op. De premium is het aantal tokens van de taal gedeeld door het aantal tokens van het Engels.
    3. Tel de bytes van elke zin in UTF-8. Tokens per byte zegt hoeveel de tokenizer van de taal leerde.
    4. Als de taal spaties tussen de woorden zet: tel de woorden die de tokenizer in meer dan één token knipt, elk woord met een spatie vooraan, zoals het woord in een zin staat.
    5. Doe hetzelfde met een andere tokenizer, op dezelfde zinnen, en vergelijk.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `C.OWN` bevat de drie zinnen van de course in zeven talen: `C.OWN["nl"][0]` is de Nederlandse zin over de bibliotheek, en `C.OWN["el"][0]` dezelfde zin in het Grieks. `C.ENC` is de tokenizer van GPT-2. De methode `encode` geeft de tokens van een tekst, en `C.ENC.decode_single_token_bytes(t)` de bytes van token `t`. Voor NTREX-128 geeft `C.counts()` de tellingen van de course over de eerste 100 zinnen van elke taal: de woorden, de tekens, de bytes, de tokens van GPT-2 en van Qwen3, en de continued words van beide, per code van de taal in NTREX, zoals `"tam"` voor het Tamil. `C.course_premium(code, tokenizer)`, `C.course_per_byte`, `C.course_continued` en `C.course_fertility` rekenen de maten uit met deze tellingen:
    """)
    return


@app.cell
def example_cell():
    _sentence = C.OWN["nl"][0]
    _tokens = C.ENC.encode(_sentence)
    print(len(_tokens), [C.ENC.decode_single_token_bytes(_t) for _t in _tokens])
    print(len(_sentence), len(_sentence.encode("utf-8")))
    _greek = C.OWN["el"][0]
    print(len(_greek), len(_greek.encode("utf-8")), len(C.ENC.encode(_greek)))
    for _code in ("nld", "tam", "zho-CN"):
        print(_code, C.counts()[_code])
        print(round(C.course_premium(_code, "gpt2"), 2), round(C.course_premium(_code, "qwen3"), 2),
              round(C.course_per_byte(_code, "gpt2"), 3), round(C.course_continued(_code, "gpt2"), 3),
              round(C.course_fertility(_code, "gpt2"), 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De Nederlandse zin is 18 tokens. *De* en *om* zijn heel, *open* ook, maar *bibliotheek* valt in vier stukjes, *b*, *ibli*, *othe* en *ek*, en *zaterdag* ook in vier. De zin heeft 47 tekens en 47 bytes: elke letter is één byte. De Griekse zin heeft 43 tekens en 79 bytes, bijna twee per teken, en GPT-2 maakt er 47 tokens van: meer tokens dan tekens. 20 van de 47 tokens bevatten de helft van een letter, één van de twee bytes, en in drie van deze tokens staat ook een spatie.

    Dan NTREX-128. De 100 Nederlandse zinnen hebben 2.391 woorden en 15.200 bytes, en GPT-2 maakt er 5.414 tokens van: een premium van 1,98, 0,356 tokens per byte, 64% van de woorden geknipt, en 2,26 tokens per woord. Het Tamil: 39.232 tokens uit 39.678 bytes, 0,989 per byte, een premium van 14,31 voor GPT-2 en 5,69 voor Qwen3; GPT-2 knipt 97% van de woorden, gemiddeld in 24,22 tokens per woord. Het Chinees kost GPT-2 3,21 keer zoveel als het Engels, en Qwen3 1,11: bijna pariteit. De 27,15 tokens per "woord" zeggen niets over het Chinees: `split` vond 324 woorden, alleen waar de tekst een spatie heeft, rond woorden in Latijnse letters, zoals *AM*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_premium():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. premium: de prijs van een taal

        Schrijf `premium`, met `texts`, een lijst met zinnen in een taal, `english`, dezelfde zinnen in het Engels, en `enc`, een tokenizer. De functie geeft de premium terug: de tokens van alle `texts` gedeeld door de tokens van alle `english`. In de startcode telt `tokens` de tokens van `texts`, met `for text in texts`, en `english_tokens` de tokens van `english`, met `for text in english`. De laatste regel deelt de twee. Vervang de twee `...` door het aantal tokens van `text`.

        De check roept `premium(["De trein naar Utrecht vertrekt over tien minuten."], ["The train to Utrecht leaves in ten minutes."], enc)` aan met de tokenizer van GPT-2 als `enc`, wat `1.5454545454545454` moet geven: 17 tokens voor de Nederlandse zin, 11 voor de Engelse. Daarna eigen gevallen: de drie zinnen van de course in een paar talen, het Engels tegenover zichzelf, en het Engels tegenover het Russisch.
        """),
        L.task("Tel het aantal tokens van elke `text` op bij de eigen telling, in beide loops."),
    ])
    return


@app.cell(hide_code=True)
def exercise_premium():
    box_premium = L.code_box("premium", lines=8)
    box_premium
    return (box_premium,)


@app.cell(hide_code=True)
def check_premium(box_premium):
    result_premium = L.check("premium", box_premium.value)
    result_premium
    return (result_premium,)


@app.cell(hide_code=True)
def hints_premium():
    L.hints(
        "premium",
        [
            "Elke loop laat de eigen telling, `tokens` of `english_tokens`, groeien met het aantal tokens van `text`: de lengte, met `len`, van de lijst met tokens die `enc` voor de tekst geeft.",
            "`enc.encode(\"Hello!\")` geeft de tokens van *Hello!*, `[15496, 0]`, en `len` van deze lijst geeft `2`.",
            "`        tokens = tokens + len(___)`",
        ],
        why="`tokens = tokens + len(enc.encode(text))`, en in de tweede loop `english_tokens = english_tokens + len(enc.encode(text))`: elke zin telt de eigen tokens mee, dus een lange zin weegt zwaarder dan een korte, net als in de tellingen van NTREX.",
    )
    return


@app.cell(hide_code=True)
def text_after_premium():
    mo.md(r"""
    In het plaatje rekent je functie de premium uit van de drie zinnen van de course in zeven talen. Het Frans kost 1,77 keer zoveel tokens als het Engels, het Nederlands 1,83 en het Duits 2,07, alle drie in het Latijnse alfabet. Het Japans kost 3,17, het Russisch 5,67 en het Grieks 6,00. Drie zinnen geven een ruwe maat: op de 100 zinnen van NTREX-128 kost het Nederlands 1,98, het Russisch 5,55 en het Grieks 6,19. En het Engels tegenover het Engels geeft precies 1: de premium vergelijkt een taal met het Engels, dus het Engels heeft per definitie pariteit.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_per_byte():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. per_byte: wat leerde de tokenizer van een taal?

        Schrijf `per_byte`, met `texts`, een lijst met zinnen, en `enc`, een tokenizer. De functie geeft de tokens van alle zinnen terug, gedeeld door de bytes in UTF-8. In de startcode beginnen `tokens` en `n_bytes` op 0, en `for text in texts` gaat langs de zinnen. Daarin telt `tokens = tokens + len(enc.encode(text))` de tokens, zoals in oefening 1. Vervang de `...` door het aantal bytes van `text` in UTF-8.

        De check roept `per_byte(["Το τρένο για την Ουτρέχτη φεύγει σε δέκα λεπτά."], enc)` aan met de tokenizer van GPT-2, wat `0.6470588235294118` moet geven: 55 tokens uit 85 bytes. Daarna eigen gevallen: de drie zinnen van de course in elk van de zeven talen, en een tekst met accenten.
        """),
        L.task("Tel het aantal bytes van `text` in UTF-8 op bij `n_bytes`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_per_byte():
    box_per_byte = L.code_box("per_byte", lines=7)
    box_per_byte
    return (box_per_byte,)


@app.cell(hide_code=True)
def check_per_byte(box_per_byte):
    result_per_byte = L.check("per_byte", box_per_byte.value)
    result_per_byte
    return (result_per_byte,)


@app.cell(hide_code=True)
def hints_per_byte():
    L.hints(
        "per_byte",
        [
            "`n_bytes` groeit met het aantal bytes van `text`: de lengte, met `len`, van de bytes die een tekst geeft met de methode `encode`, in UTF-8.",
            "`\"één\".encode(\"utf-8\")` geeft de bytes van *één*: twee voor elke *é* en één voor de *n*, dus `len` daarvan geeft `5`, terwijl `len(\"één\")` 3 tekens telt.",
            "`        n_bytes = n_bytes + len(___)`",
        ],
        why="`n_bytes = n_bytes + len(text.encode(\"utf-8\"))`: `len(text)` telt tekens, en een Griekse letter is één teken en twee bytes. `text.encode()` zonder argument geeft hetzelfde, omdat Python UTF-8 gebruikt als je niets anders zegt.",
    )
    return


@app.cell(hide_code=True)
def text_after_per_byte():
    mo.md(r"""
    In het plaatje je functie op de zinnen van de course: het Engels 0,222 tokens per byte, het Frans 0,356, het Nederlands 0,357, het Duits 0,392, het Japans 0,528, het Russisch 0,603 en het Grieks 0,604. De Griekse zin over de bibliotheek laat zien waarom: een gestippeld token bevat een deel van een letter, en veel andere tokens zijn één letter. Op de 100 zinnen van NTREX-128 maakt GPT-2 0,989 tokens per byte van het Tamil, 0,992 van het Telugu en 0,985 van het Armeens: bijna geen stukjes van deze talen. Qwen3 maakt 0,397 tokens per byte van het Tamil.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_continued():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. continued: hoe vaak knipt de tokenizer een woord?

        Schrijf `continued`, met `texts`, een lijst met zinnen, en `enc`, een tokenizer. De functie geeft het deel van de woorden terug dat de tokenizer in meer dan één token knipt. In de startcode telt `words` alle woorden en `count` de geknipte woorden. `for text in texts` gaat langs de zinnen, en `for word in text.split()` langs de woorden van elke zin. Daarin telt `words = words + 1` het woord, en `count = count + 1` loopt als de test na `if` klopt. Vervang de `...` door de test: knipt de tokenizer `word` in meer dan één token? Geef het woord een spatie vooraan, zoals het woord in een zin staat: GPT-2 heeft andere tokens voor een woord aan het begin van een tekst.

        De check roept `continued(["De trein naar Utrecht vertrekt over tien minuten."], enc)` aan met de tokenizer van GPT-2, wat `0.75` moet geven: zes van de acht woorden zijn geknipt. Daarna eigen gevallen: de drie zinnen van de course in elk van de zeven talen, en korte woorden.
        """),
        L.task("Test of `word`, met een spatie vooraan, meer dan één token wordt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_continued():
    box_continued = L.code_box("continued", lines=9)
    box_continued
    return (box_continued,)


@app.cell(hide_code=True)
def check_continued(box_continued):
    result_continued = L.check("continued", box_continued.value)
    result_continued
    return (result_continued,)


@app.cell(hide_code=True)
def hints_continued():
    L.hints(
        "continued",
        [
            "Zet een spatie voor `word`, codeer het geheel met `encode` van `enc`, en test met `len` of de lijst met tokens meer dan 1 token bevat.",
            "`+` plakt twee teksten aan elkaar: `\"a\" + \"b\"` geeft `\"ab\"`. En de spatie telt mee: `enc.encode(\"leaves\")` geeft twee tokens, *le* en *aves*, en `enc.encode(\" leaves\")` geeft er één.",
            "`            if len(enc.encode(___)) > 1:`",
        ],
        why="`if len(enc.encode(\" \" + word)) > 1:`: in een zin staat er voor elk woord behalve het eerste een spatie, en GPT-2 leerde de meeste woorden met deze spatie. Een woord van precies één token is heel; vanaf twee tokens is het woord geknipt.",
    )
    return


@app.cell(hide_code=True)
def text_after_continued():
    mo.md(r"""
    In het plaatje je functie op de zinnen van de course: GPT-2 knipt 16% van de Engelse woorden, 62% van de Franse, 70% van de Nederlandse en 75% van de Duitse, en elk Russisch en Grieks woord. Het Japans krijgt ook 100%, maar dat zegt niets: `split` vindt geen spatie in een Japanse zin, dus elke zin is één "woord". Onder de tabel staan de woorden van de Nederlandse zin over de bibliotheek, met een punt tussen de stukjes. Op NTREX-128 knipt Qwen3 44% van de Nederlandse woorden, tegen 64% bij GPT-2.

    ### Probeer het: één zin in 48 talen

    De demo heeft de eerste 100 zinnen van NTREX-128 in 48 talen nodig, en de tokenizer van Qwen3. Kies een taal en een tokenizer: het plaatje knipt de eerste zin van NTREX-128 in tokens, in het Engels en in de gekozen taal, en de tabel geeft de maten van deze les over de eerste 100 zinnen, voor beide tokenizers.

    Begin met het Tamil. Probeer dan het Vietnamees met beide tokenizers, en kijk naar de continued words; dan het Japans, en het Yoruba, in het Latijnse alfabet geschreven, met accenten.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label=f"Download NTREX-128 en de tokenizer van Qwen3, {C.download_size(C.FILES, L.lang)}")
    return (get_button,)


@app.cell(hide_code=True)
def ntrex_cell(get_button):
    try_languages = C.files_or_stop(get_button, {
        "ask": f"*De zinnen van NTREX-128 en de tokenizer van Qwen3 staan nog niet op je computer. De knop downloadt de bestanden één keer, {C.download_size(C.FILES, L.lang)}: de zinnen van de repository van NTREX op GitHub, onder de licentie CC BY-SA 4.0, en de tokenizer van Hugging Face, onder de licentie Apache-2.0. De rest van de les werkt zonder.*",
        "busy": "**Bezig met downloaden** {done} van {total} MB …",
        "no_net": "*Het downloaden lukte niet: GitHub of Hugging Face was niet bereikbaar, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*Het downloaden lukte niet: een binnengekomen bestand is niet het vastgelegde bestand van de course, dus de course gooide het bestand weg. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*Het downloaden lukte niet: er is op je schijf geen plek voor de bestanden. Maak ruimte, en klik opnieuw.*",
        "failed": "*Het downloaden lukte niet ({why}). Klik opnieuw.*",
    }, L.lang)
    try_language = mo.ui.dropdown(options=try_languages, value=C.ntrex.name(C.TRY_START, L.lang), label="Taal")
    try_tokenizer = mo.ui.radio(options={"GPT-2": "gpt2", "Qwen3": "qwen3"}, value="GPT-2", label="Tokenizer", inline=True)
    mo.hstack([try_language, try_tokenizer], justify="start", gap=1.5, wrap=True)
    return try_language, try_tokenizer


@app.cell(hide_code=True)
def try_look(try_language, try_tokenizer):
    mo.Html(C.try_picture(L.lang, try_language.value, try_tokenizer.value))
    return


@app.cell(hide_code=True)
def text_after_try():
    mo.md(r"""
    Het Tamil met GPT-2: bijna elk token van de zin is gestippeld, één byte van een letter. Met Qwen3 zijn veel tokens hele letters, of meer letters, en de premium zakt van 14,31 naar 5,69. Het Vietnamees staat in het Latijnse alfabet, met veel accenten, en zet een spatie na elke lettergreep. Voor GPT-2 kost het Vietnamees 4,55 keer zoveel als het Engels, en GPT-2 knipt 93% van de "woorden"; voor Qwen3 kost het 1,45 keer zoveel, en Qwen3 knipt 19%, ongeveer even vaak als Engelse woorden. In de 100 Japanse zinnen vindt `split` 125 "woorden", alleen waar een spatie staat, dus de 57 tokens per woord van GPT-2 zeggen niets over het Japans; de premium, 2,62 voor GPT-2 en 1,33 voor Qwen3, vergelijkt wel goed. Het Yoruba geeft de tonen aan met accenten boven de letters, en een paar klanken met een punt onder de letter: een letter met een accent is twee bytes of meer, en de premium van GPT-2 is 5,38. Bij Qwen3 is de premium 3,24, meer dan bij elke andere taal in het Latijnse alfabet in de tabel.

    ### Oefening 4. Helpt het of schiet het tekort?

    Een taal heeft weinig tekst en bijna geen gelabelde data. Deel elke aanpak hieronder in: helpt de aanpak zo'n taal aan een model dat werkt, of schiet de aanpak tekort?
    """)
    return


@app.cell(hide_code=True)
def exercise_helps():
    sort_helps = L.sort("helps")
    sort_helps
    return (sort_helps,)


@app.cell(hide_code=True)
def check_helps(sort_helps):
    result_helps = L.check("helps", sort_helps)
    result_helps
    return (result_helps,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De tokens komen uit [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl), en de groepen van talen uit [Zo'n 7.000 talen](/course/open/nlp-languages-of-the-world?lang=nl). Cross-lingual transfer is de fine-tuning van [Een encoder finetunen voor een taak](/course/open/nlp-fine-tune-bert?lang=nl), in de ene taal voor de andere. Je mat de premium, tokens per byte en de continued words, en zag dat een tokenizer voor veel talen de prijs verlaagt, maar niet tot pariteit. De volgende les, *Modellen voor het Nederlands*, gaat over modellen voor één taal uit groep 4: het Nederlands.

    **Een voorbeeld uit het dagelijks leven.** Reis je naar een land zonder eigen taalgids, dan neem je de taalgids van een buurtaal mee: veel daarvan werkt, net goed genoeg. Een paar zinnetjes van de mensen ter plaatse doen de rest.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_premium, result_per_byte, result_continued, result_helps):
    L.wrap_up(result_premium, result_per_byte, result_continued, result_helps)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
