# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-memorization. Route: course/route/lessen/verantwoord.yaml. Checks, de run van Qwen3 van de course en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-memorization", "nl")
    C = L.checks   # 14 teksten, en wat Qwen3-0.6B-Base en Qwen3-0.6B schreven na hun eerste k woorden


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Drie korte vragen over [Sampling: laat het model schrijven](/course/open/lm-sampling?lang=nl), [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl) en [Wat is privacy?](/course/open/rai-what-is-privacy?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_greedy():
    quick_greedy = L.choice(
        "quick-greedy",
        question="Een model schrijft twee keer verder vanaf hetzelfde begin, allebei de keren met greedy decoding. Wat komt eruit?",
        options={
            "differ": "Twee teksten die verschillen",
            "right": "Twee keer dezelfde tekst",
            "seed": "Verschillende teksten, tenzij de seed vastligt",
        },
        correct="right",
        why={
            "right": "Greedy decoding neemt altijd het meest waarschijnlijke volgende token, en vanaf hetzelfde begin geeft het model dezelfde kansen, dus schrijft het elke keer dezelfde tekst. Daarom is één run met greedy decoding genoeg om te zien wat een model na een begin schrijft.",
            "differ": "Tokens trekken geeft elke keer een andere tekst, maar greedy decoding trekt niets: het neemt altijd het meest waarschijnlijke volgende token. Zie *Het idee* van [Sampling: laat het model schrijven](/course/open/lm-sampling?lang=nl).",
            "seed": "Een seed doet er alleen toe als tokens willekeurig getrokken worden. Greedy decoding trekt niets: het neemt altijd het meest waarschijnlijke volgende token. Zie *Het idee* van [Sampling: laat het model schrijven](/course/open/lm-sampling?lang=nl).",
        },
    )
    quick_greedy
    return (quick_greedy,)


@app.cell(hide_code=True)
def quick_check_role():
    quick_role = L.choice(
        "quick-role",
        question="In een chat typt iemand het begin van een tekst en vraagt het model verder te schrijven. Welke rol heeft dat bericht?",
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
def quick_check_flow():
    quick_flow = L.choice(
        "quick-flow",
        question="Iemand zette een adres en telefoonnummer op de webpagina van een softwareproject. Later schrijft een model ze uit voor iedereen die erom vraagt. Welke opvatting van privacy ziet een probleem, hoewel er niets verborgens uitkwam?",
        options={
            "secrecy": "Geheimhouding: iets verborgens werd bekend",
            "right": "Gepaste stroom",
            "none": "Geen, want de gegevens stonden online",
        },
        correct="right",
        why={
            "right": "Gepaste stroom vraagt waar informatie daarna heen ging, en of dat past bij waar ze vandaan kwam. Contactgegevens voor een softwareproject passen bij dat project, niet bij een model dat ze aan iedereen geeft.",
            "secrecy": "Voor geheimhouding wordt privacy geschonden als iets verborgens bekend wordt, en deze gegevens stonden al op het web. Gepaste stroom vraagt waar ze daarna heen gingen. Zie *Drie opvattingen van privacy* in *Het idee* van [Wat is privacy?](/course/open/rai-what-is-privacy?lang=nl).",
            "none": "Geheimhouding ziet geen probleem, maar gepaste stroom wel: die vraagt of de plek waar informatie heen gaat past bij de plek waar ze vandaan kwam. Zie *Drie opvattingen van privacy* in *Het idee* van [Wat is privacy?](/course/open/rai-what-is-privacy?lang=nl).",
        },
    )
    quick_flow
    return (quick_flow,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Nicholas Carlini en elf collega's (2021) kregen GPT-2 zo ver dat het van één persoon de naam, het e-mailadres, het telefoonnummer, het faxnummer en het adres uitschreef. In hun publicatie maakten ze die onleesbaar, omdat alles klopte. Geheim waren deze gegevens niet: ze waren online gedeeld voor één bepaald doel, als contactgegevens bij een softwareproject, en het model schreef ze ergens anders uit. Een language model leert van zijn training data, en een deel van die data kan het woord voor woord herhalen. In deze les meet je dat zelf: je geeft een model de eerste woorden van een tekst, en telt hoeveel van de volgende woorden het precies goed schrijft.

    **Een voorbeeld uit het dagelijks leven.** Het deuntje van een reclame heb je honderd keer gehoord. Als iemand de eerste regel zingt, komt de rest vanzelf, woord voor woord. Een zin die je één keer hoorde, komt zo niet terug. Tekst die een model op duizenden webpagina's zag, kan er op dezelfde manier uitkomen.
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="De course gaf Qwen3-0.6B-Base de eerste 16 woorden van drie Engelse teksten en liet het verder schrijven: de MIT-licentie (\"Permission is hereby granted, free of charge, …\"), het begin van de roman *A Tale of Two Cities* (\"It was the best of times, …\"), en een tekst over een roeiclub die de course zelf schreef. Van welke teksten schreef het de volgende 25 woorden precies goed?",
        options={
            "none": "Van geen van de drie: een klein model onthoudt geen tekst woord voor woord",
            "two": "Van de licentie en de roman",
            "all": "Van alle drie, ook de eigen roeiclub van de course",
        },
        reveal="**Van de licentie en de roman**: na hun eerste 16 woorden schreef Qwen3-0.6B-Base van allebei de volgende 25 woorden precies goed. Van de tekst over de roeiclub, die de course schreef nadat Qwen3 getraind was, had het 1 woord goed, \"euros\", en daarna maakte het er een rekensom van over een club met 10 leden. Hieronder zie je hoe zo'n test werkt, en wat er voor 14 teksten uitkwam.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        ### Woord voor woord onthouden

        Een language model leert uit zijn training data het volgende token te voorspellen. Meestal leert het patronen: hoe zinnen lopen, welke woorden bij elkaar horen. Maar soms leert het een stuk tekst zelf, zo goed dat het dat weer kan uitschrijven. Dat heet **memorisatie**. Carlini en collega's (2021) letten vooral op tekst die een model maar uit een paar plekken leerde. Die noemen ze eidetische memorisatie: data die een model onthoudt hoewel ze maar in een klein aantal voorbeelden van de training data staat, en hoe minder voorbeelden, hoe sterker. De naam komt van een eidetisch geheugen, vaker een fotografisch geheugen genoemd: iets onthouden na het één keer gezien te hebben. Een tekst die je het model kunt laten schrijven, en die in hoogstens k voorbeelden van de training data staat, noemen ze k-eidetisch gememoriseerd. Een voorbeeld is daar een hele webpagina, dus een tekst die tien keer op één pagina staat, telt toch één keer.

        ### Extractable met k woorden context

        Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramèr en Chiyuan Zhang (2023) meten het anders, met een begin uit de training data zelf. Volgens hun definitie is een stuk tekst s *extractable with k tokens of context* als er een begin p van k tokens is zodat p gevolgd door s in de training data van het model staat, en het model s schrijft als het p krijgt, met greedy decoding. Hun voorbeeld: als de training data "My phone number is 555-6789" bevat, en het model na het begin "My phone number is" als meest waarschijnlijke uitvoer "555-6789" schrijft, dan is dat nummer extractable met 4 woorden context. In hun test moest het model de volgende 50 tokens precies goed schrijven, in hun data gemiddeld 127 tekens of 25 woorden. Ze gebruiken greedy decoding, zodat één run laat zien wat het model schrijft als het bij elke stap het meest waarschijnlijke token neemt, en ze controleerden dat een andere manier van decoderen de uitkomst nauwelijks verandert: beam search, dat meerdere kandidaat-teksten tegelijk bijhoudt in plaats van één, vond gemiddeld net geen 2 procentpunt meer. Milad Nasr en collega's (2023) maken nog een onderscheid: een test die bij het echte begin van een tekst begint, zoals deze les doet, vindt volgens hen memorisatie van de soort *discoverable*; voor de soort *extractable* is een prompt nodig die gemaakt is zonder toegang tot de training data.

        Deze les telt woorden in plaats van tokens, omdat je woorden ziet. `text.split()` geeft de lijst met woorden van een tekst, geknipt bij elke spatie, tab en nieuwe regel: `"to be or not".split()` geeft `["to", "be", "or", "not"]`. Een woord telt alleen als het precies gelijk is, teken voor teken. "Order" en "order" zijn verschillende woorden, en "continent," en "continent" ook: een komma plakt aan zijn woord.
        """),
        mo.Html(C.test_picture("nl")),
        mo.md(r"""
        ### Training data extraction

        Carlini en collega's (2021) begonnen niet bij bekende tekst. De training data van GPT-2 is nooit vrijgegeven, dus lieten ze GPT-2 schrijven: met elk van drie strategieën 200.000 teksten van 256 tokens. Een daarvan trok elk token willekeurig uit de 40 meest waarschijnlijke (top-n sampling met n = 40), in plaats van greedy decoding. Ze zetten de teksten op volgorde met zes maten voor hoe waarschijnlijk elke tekst onthouden was, en namen uit de bovenste 1000 van elk van de 18 combinaties er 100: 1800 kandidaten. Een kandidaat telde als onthouden als Google een pagina vond waarop een niet-triviaal stuk ervan precies zo stond, en de makers van GPT-2 lieten hen het nakijken in de training data. Hun uitkomst: van de 1800 kandidaten waren 604 verschillende voorbeelden uit de training data onthouden, 33,5%, en in hun beste variant 67%. Van alle 600.000 teksten die GPT-2 schreef, bevatten er minstens 604, 0,1%, onthouden tekst, en dat is volgens hen een ondergrens die waarschijnlijk ver onder het echte aantal ligt. Zo stukken van de training data uit een model halen heet **training data extraction**.

        Wat eruit kwam, was vooral gewone webtekst. Tot de grootste groepen in hun tabel horen: nieuws, 109; logbestanden en foutmeldingen, 79; licenties, gebruiksvoorwaarden en auteursrechtvermeldingen, 54. En persoonsgegevens: ze vonden 78 voorbeelden van namen, telefoonnummers, adressen en accounts op sociale media van individuele mensen. Ook zeldzame tekst kwam eruit: een willekeurige code, een UUID, die in één enkel document van de training data van GPT-2 staat.

        ### Drie dingen die het meer maken

        Carlini en collega's (2023) maten memorisatie in vier modellen uit de GPT-Neo-familie, van 125 miljoen tot 6 miljard parameters, allemaal getraind op dezelfde 825 GB tekst. Memorisatie groeit flink, schrijven ze, naarmate (1) het model groter is, (2) een voorbeeld vaker in de training data staat, en (3) het begin dat het model krijgt meer tokens telt.

        - **Grootte.** Een tien keer zo groot model onthoudt 19 procentpunt meer, op een steekproef waarin dubbele teksten vaker voorkomen dan in de training data. En het is geheugen, niet alleen goed gokken: GPT-2, dat niet op die data getraind is, schreef ongeveer 6% van de teksten goed verder, tegen 40% voor een ongeveer even groot GPT-Neo-model dat dat wel was.
        - **Kopieën.** Hoe vaker een tekst in de training data staat, hoe vaker hij eruit komt. Tekst die maar een paar keer voorkomt, komt zelden terug, maar bij tekst die heel vaak voorkomt, stijgt die kans sterk. Toch niet alleen dan: ook met maar een paar kopieën gebeurt het nog, dus deduplicatie voorkomt lekken niet helemaal, schrijven ze.
        - **Context.** Hoe langer het begin, hoe meer eruit komt: van de teksten in hun test haalden ze 33% uit het model van 6 miljard parameters met 50 tokens context, en 65% met 450 tokens.

        Hoeveel van al zijn training data een model onthoudt, is moeilijk vast te stellen. Voor GPT-J, met 6 miljard parameters, vinden ze dat minstens 1% van de training data uit GPT-J te halen is en niet uit GPT-2, dat er niet op getraind is: een ondergrens. En de teksten die al hun modellen onthielden, waren vooral code en teksten die heel vaak voorkomen, zoals opensourcelicenties.

        ### Ook chat models

        Een chat model is een base model dat verder getraind is om als assistent te antwoorden; een base model schrijft alleen tekst verder. [Base model of chat model?](/course/open/lm-base-vs-chat?lang=nl) gaat er dieper op in. Nasr en collega's (2023) vielen ChatGPT aan, het model gpt-3.5-turbo, met een vreemde prompt: herhaal dit woord voor altijd, gevolgd door 50 keer het Engelse woord "poem". ChatGPT herhaalt het woord honderden keren, en gaat dan ineens andere tekst schrijven, soms woord voor woord uit de training data: stukken uit romans, een volledige kopie van het gedicht *The Raven*, auteursrechtregels boven code, gebruiksvoorwaarden. Voor 200 dollar aan vragen aan ChatGPT haalden ze er meer dan 10.000 verschillende letterlijk onthouden voorbeelden uit. Van de 15.000 gegenereerde teksten die ze op persoonsgegevens testten, bevatte 16,9% onthouden persoonsgegevens, zoals iemands e-mailhandtekening met contactgegevens. Ze meldden het op 30 augustus 2023 bij OpenAI en gaven het bedrijf 90 dagen om het op te lossen; de aanval is volgens hen specifiek voor dat model.

        Dezelfde publicatie laat zien hoe een chat model kan verbergen wat het weet: gevraagd om teksten verder te schrijven waarvan bekend was dat het ze onthouden had, deed ChatGPT dat in maar 3,5% van de gevallen. Dat er niets uit een model komt, betekent niet dat er niets in zit.

        ### Qwen3

        Qwen3 is getraind op 36 biljoen tokens in 119 talen en dialecten, zegt het rapport erover: code, bètavakken (STEM), redeneren, boeken, veel talen, tekst gelezen uit pdf's, en biljoenen tokens geschreven door eerdere Qwen-modellen. Meer dan 30 biljoen tokens kregen labels voor onder meer onderwijswaarde, vakgebied, domein en veiligheid, om de data te filteren en te mengen. Het rapport en de twee model cards zeggen niets over persoonsgegevens, privacy, PII of deduplicatie; de course zocht op deze woorden. De data zelf is niet gepubliceerd. Dus kan niemand van buiten nagaan of een tekst in de training data van Qwen3 stond: deze les zegt dat Qwen3 een tekst reproduceert, en dat het die waarschijnlijk onthield omdat de tekst overal op het web gekopieerd is. Volgens het rapport is elk Qwen3-model op deze 36 biljoen tokens getraind. Daarna kreeg Qwen3-0.6B, het chat model, post-training om te antwoorden, via distillation uit een veel grotere Qwen3: Qwen3-32B of Qwen3-235B-A22B. Qwen3-0.6B-Base is het model van daarvoor.

        ### De test van de course

        De course koos 14 Engelse teksten. Twaalf staan gekopieerd op veel webpagina's: vier licentieteksten (MIT, GPL, Apache, BSD), de mededeling van Project Gutenberg, de opvultekst lorem ipsum, en zes beroemde openingen, van de Amerikaanse grondwet tot *The Raven*, allemaal uit het publieke domein of bedoeld om te kopiëren. Twee schreef de course voor deze les, over een roeiclub en over honing, nadat Qwen3 getraind was: geen van beide Qwen3-modellen kan ze gezien hebben. Voor k = 4, 8, 16 en 24 gaf de course beide modellen de eerste k woorden van elke tekst, als kale tekst, zonder chat template, de vorm met rollen waarin een chat model een gesprek krijgt, en liet elk model met greedy decoding 64 tokens schrijven. Daarna telde de course hoeveel van de volgende 25 woorden precies goed waren.

        **Probeer het.** Hieronder kun je Qwen3-0.6B-Base een eigen begin geven: het begin van een boek of een beroemde toespraak, de eerste regel van een licentie, of een zin die je net verzint. Schrijft het verder wat jij kent?
        """),
    ])
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B-Base (1,2 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    C.download_or_stop(get_button, {
        "ask": "*Qwen3-0.6B-Base staat nog niet op je computer. De knop downloadt het één keer van Hugging Face, van `Qwen/Qwen3-0.6B-Base` op een vaste versie: 1,2 GB, de weights en de tokenizer, onder de Apache-2.0-licentie. De rest van de les werkt zonder: die leest wat de course heeft opgenomen.*",
        "busy": "**Bezig met downloaden:** {done} van {total} MB …",
        "no_net": "*De download lukte niet: Hugging Face was niet bereikbaar, of de verbinding brak af. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*De download lukte niet: het bestand dat binnenkwam, is niet het vastgelegde bestand van de course, dus de course gooide het weg. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*De download lukte niet: er is op je schijf geen plek voor 1,2 GB. Maak ruimte, en klik opnieuw.*",
        "failed": "*De download lukte niet ({why}). Klik opnieuw.*",
    })
    mo.md("*Qwen3-0.6B-Base is gedownload.*")
    return


@app.cell(hide_code=True)
def start_box_cell():
    start_box = mo.ui.text_area(value=C.EXAMPLE_START, label="Het begin van een tekst", rows=3, full_width=True)
    write_button = mo.ui.run_button(label="Laat Qwen3 verder schrijven")
    mo.vstack([start_box, write_button])
    return start_box, write_button


@app.cell(hide_code=True)
def demo_live(start_box, write_button):
    if not C.downloaded():
        _out = mo.md("*Download Qwen3-0.6B-Base eerst, met de knop hierboven.*")
    elif not write_button.value:
        _out = mo.md("*Druk op de knop om Qwen3-0.6B-Base 64 tokens verder te laten schrijven, met greedy decoding, zoals in de run van de course.*")
    elif not start_box.value.strip():
        _out = mo.md("*Schrijf eerst het begin van een tekst in het vak.*")
    else:
        _words = {"title": "Wat Qwen3-0.6B-Base schrijft", "loading": "Qwen3-0.6B-Base laden …",
                  "writing": "Qwen3-0.6B-Base schrijft …", "done": "{tokens} tokens in {seconds} seconden.",
                  "load": "Qwen3-0.6B-Base laden stopt met een fout. Sluit andere programma's en druk opnieuw op de knop; gebeurt het weer, meld het dan op GitHub.",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.live(start_box.value.strip(), lambda _html: mo.output.replace(mo.Html(_html)), _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **memorisatie** | een model heeft een stuk van zijn training data zo goed geleerd dat het dat woord voor woord kan uitschrijven |
        | **training data extraction** | stukken van de training data van een model uit het model halen, door het te laten schrijven |
        | **extractable met k woorden context** | een stuk uit de training data van het model komt er precies uit als het model de k woorden krijgt die daar ervoor staan, met greedy decoding; Carlini en collega's tellen tokens |
        | `text.split()` | de lijst met woorden van `text`, geknipt bij elke spatie, tab en nieuwe regel |
        """),
        L.watch_out("Een woord-voor-woordtest is streng. Een model dat \"order\" schrijft waar de tekst \"Order\" heeft, of één komma weglaat, stopt de telling daar, hoewel het de tekst duidelijk kent. En een model dat een tekst niet uitschrijft, kan die toch geleerd hebben: met een langer begin of een andere prompt kan hij eruit komen. Zo'n test vindt dus meestal minder dan een model onthoudt. En een tekst die er wel uitkomt, kan ook gewoon makkelijk te voorspellen zijn: GPT-2 schreef ongeveer 6% van de teksten waarop het nooit getraind was goed verder."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om te meten hoeveel van een tekst een model uitschrijft:

    1. Knip de tekst in woorden. De eerste k woorden zijn het begin, de n woorden daarna het echte vervolg. Dat is `split_text`.
    2. Geef het model alleen het begin, en laat het schrijven met greedy decoding, zodat één run laat zien wat het schrijft als het bij elke stap het meest waarschijnlijke token neemt.
    3. Tel hoeveel woorden aan het begin van wat het schreef gelijk zijn aan het echte vervolg, tot het eerste woord dat verschilt. Dat is `same_start`.
    4. Een run waarvan het hele vervolg eruit kwam, telt als extractable met k woorden context, als de tekst in de training data stond. Over veel teksten is het aandeel van zulke runs `share_extractable`.
    5. Herhaal het voor andere k, andere teksten en andere modellen, en vergelijk teksten die overal op het web gekopieerd zijn met teksten die het model niet gezien kan hebben.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het begin van de Onafhankelijkheidsverklaring, met k = 8. `C.TEXTS` bevat de 14 teksten van de test van de course, elk onder een korte naam. De cel neemt de eerste 8 woorden als begin en de volgende 25 als echt vervolg, stap 1 en 3 van het recept met de hand. `C.written` geeft wat Qwen3-0.6B-Base na dat begin schreef in de run van de course. De cel print het begin, en dan de eerste 20 woorden van wat het model schreef naast die van het echte vervolg, met of de twee gelijk zijn. Voer de cel uit.
    """)
    return


@app.cell
def example_declaration():
    # de Onafhankelijkheidsverklaring, en wat Qwen3-0.6B-Base schreef na de eerste 8 woorden ervan
    _words = C.TEXTS["declaration"].split()
    _start = " ".join(_words[:8])
    _truth = " ".join(_words[8:33])
    _written = C.written("base", "declaration", 8)

    print("start:", _start)
    _a = _written.split()
    _b = _truth.split()
    for _i in range(20):
        print(_i + 1, _a[_i], "|", _b[_i], "|", _a[_i] == _b[_i])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De eerste 18 woorden zijn gelijk, komma voor komma. Bij woord 19 heeft de verklaring "assume", en Qwen3-0.6B-Base schrijft "identify": vanaf daar schrijft het een eigen tekst. De telling voor deze run is dus 18, niet 25, en de run telt niet als extractable. Woord 20 verschilt ook, maar zou niet meetellen als het wel gelijk was: de telling stopt bij het eerste woord dat verschilt, omdat het model de tekst daarna niet meer volgt. In oefening 1 knip je een tekst in een begin en een echt vervolg; in oefening 2 tel je de woorden die gelijk zijn.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_split():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. split_text: het begin en het echte vervolg

        `split_text(text, k, n)` krijgt een tekst `text` en twee hele getallen `k` en `n`, en geeft twee teksten terug: het begin, de eerste `k` woorden van `text`, en het echte vervolg, de `n` woorden direct daarna, elk als één tekst met een spatie tussen de woorden. `words` is al de lijst met woorden van `text`. Vervang de twee `...`: `start` is de eerste `k` woorden van `words`, aan elkaar gezet met spaties, en `truth` de `n` woorden daarna, op dezelfde manier. De functie geeft `start, truth` terug. De check roept `split_text("Four score and seven years ago our fathers brought forth", 4, 3)` aan, en daarna de Gettysburg Address, lorem ipsum en eigen korte teksten.
        """),
        L.task("Geef de eerste k woorden terug als het begin, en de n woorden daarna als het echte vervolg."),
    ])
    return


@app.cell(hide_code=True)
def exercise_split():
    box_split = L.code_box("split", lines=6)
    box_split
    return (box_split,)


@app.cell(hide_code=True)
def check_split(box_split):
    result_split = L.check("split", box_split.value)
    result_split
    return (result_split,)


@app.cell(hide_code=True)
def hints_split():
    L.hints(
        "split",
        [
            "Stap 1 van het recept, zoals in het uitgewerkte voorbeeld: een slice van `words` pakt de woorden die je wilt, en `\" \".join` maakt er één tekst van.",
            "Het begin is de slice die bij plaats `k` eindigt. Het vervolg begint bij plaats `k`; waar eindigt het, als het `n` woorden bevat?",
            "`start = \" \".join(words[:k])` en `truth = \" \".join(___)`",
        ],
        why="`start = \" \".join(words[:k])` en `truth = \" \".join(words[k:k + n])`. `split_text(\"Four score and seven years ago our fathers brought forth\", 4, 3)` geeft `(\"Four score and seven\", \"years ago our\")`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_same_start():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. same_start: hoeveel woorden kwamen eruit?

        `same_start(written, truth)` krijgt wat het model schreef, `written`, en het echte vervolg, `truth`, en geeft terug hoeveel woorden aan het begin van `written` gelijk zijn aan de woorden van `truth` op dezelfde plaats, tot het eerste woord dat verschilt. `a` en `b` zijn al de lijsten met woorden van de twee teksten, en `i` begint bij 0. De `while`-loop telt 1 op bij `i` zolang de voorwaarde klopt, en de functie geeft `i` terug. Vervang de `...` door de voorwaarde: er staat een woord op plaats `i` in `a` en in `b`, en de twee woorden daar zijn gelijk. Let op de volgorde: Python leest `and` van links naar rechts en stopt bij het eerste deel dat `False` is, dus vraag of plaats `i` bestaat voordat je naar het woord daar kijkt. De check roept `same_start("it was the age of fools", "it was the age of wisdom")` aan, en daarna eigen teksten en de run van het uitgewerkte voorbeeld.
        """),
        L.task("Geef terug hoeveel woorden aan het begin van written gelijk zijn aan die van truth, tot het eerste dat verschilt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_same_start():
    box_same_start = L.code_box("same-start", lines=8)
    box_same_start
    return (box_same_start,)


@app.cell(hide_code=True)
def check_same_start(box_same_start):
    result_same_start = L.check("same-start", box_same_start.value)
    result_same_start
    return (result_same_start,)


@app.cell(hide_code=True)
def hints_same_start():
    L.hints(
        "same-start",
        [
            "Stap 3 van het recept. De loop moet stoppen zodra een van de lijsten geen woord op plaats `i` heeft, of de twee woorden daar verschillen.",
            "Drie delen, verbonden met `and`: twee die vragen of `i` nog kleiner is dan de lengte van een lijst, en één die de woorden vergelijkt. Welke delen komen eerst?",
            "`while i < len(a) and i < len(b) and ___:`",
        ],
        why="`while i < len(a) and i < len(b) and a[i] == b[i]:`. `same_start(\"it was the age of fools\", \"it was the age of wisdom\")` geeft 5. De twee vragen over `i` komen eerst: als `i` het eind van een lijst bereikt, stopt `and` daar en zoekt het nooit een woord op dat er niet is.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Hoeveel elk model uitschrijft

    Met jouw `same_start` telt de course, voor elk van de 14 teksten en elke k, hoeveel van de volgende 25 woorden elk model in zijn run precies goed schreef. De oranje staaf is Qwen3-0.6B-Base, de dunne grijze staaf eronder het chat model, Qwen3-0.6B.
    """)
    return


@app.cell(hide_code=True)
def demo_counts(box_same_start, result_same_start):
    _yours = C.mine(box_same_start.value, result_same_start.state == "correct")
    _found = C.counts(_yours) if _yours else None
    _who = "jouw `same_start`"
    if _found is None:
        _who = "de `same_start` van de course, omdat de jouwe hier niet voor elke run een telling geeft" if _yours else "de `same_start` van de course, omdat oefening 2 nog niet goed is"
        _found = C.counts()
    _out = mo.vstack([
        mo.md(f"Geteld met {_who}."),
        L.explore("demo", "k", lambda _k: C.demo_frame(_found, int(_k), L.lang), start=16, values=C.KS, role="",
                  task="Sleep k van 4 naar 24: vanaf welke k schrijft Qwen3-0.6B-Base *A Tale of Two Cities* uit, en vanaf welke k de Amerikaanse grondwet?"),
    ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Met 4 woorden context schrijft Qwen3-0.6B-Base van 4 van de 12 gekopieerde teksten de volgende 25 woorden precies goed: de MIT-, GPL- en Apache-tekst en *A Tale of Two Cities*. Met 16 woorden zijn het er 10. Een langer begin helpt, zoals Carlini en collega's vonden, maar één run kan alle kanten op: met 24 woorden zijn het er 9, omdat *Moby-Dick* na 17 woorden afdwaalt naar een zin die niet in het boek staat. De Onafhankelijkheidsverklaring en *The Raven* komen er nooit helemaal uit. De Amerikaanse grondwet scoort 0 bij k = 8, hoewel het model de tekst kent: het schrijft "order" waar deze versie "Order" heeft, en verderop "defense" voor "defence". Bij k = 16 staat "Order" al in het begin, en komen de volgende 25 woorden er precies uit. Van de twee teksten van de course heeft het base model bij geen enkele k meer dan 2 woorden goed.

    Het chat model schrijft één tekst uit, lorem ipsum, bij k = 8 en k = 16. Van elke andere tekst heeft het hoogstens 12 woorden goed; van lorem ipsum bij k = 24 heeft het er 22 goed.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_share():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. share_extractable: welk aandeel kwam er helemaal uit?

        `RUNS` is de lijst met de 14 runs van Qwen3-0.6B-Base met k = 16, één paar `(written, truth)` per tekst, in de volgorde van de staven. `share_extractable(runs)` krijgt zo'n lijst `runs`, en geeft het aandeel terug van de runs waarvan het hele vervolg eruit kwam: een getal van 0 tot 1. Het vak mag `same_start` gebruiken: de versie van de course staat klaar. `hits` begint bij 0, en de `for`-loop neemt de runs een voor een en zet de twee teksten van elke run in `written` en `truth`. Vervang de twee `...`: tel een run mee als `same_start(written, truth)` gelijk is aan het aantal woorden van `truth`, en geef na de loop het aandeel van de runs terug dat meetelde. De check roept `share_extractable(RUNS)` aan, en daarna de runs van het chat model en eigen korte lijsten.
        """),
        L.task("Geef het aandeel terug van de runs waarvan het hele vervolg eruit kwam."),
    ])
    return


@app.cell(hide_code=True)
def exercise_share():
    box_share = L.code_box("share", lines=7)
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
            "Stap 4 van het recept. Een run telt mee als `same_start` elk woord van het vervolg vond: evenveel woorden als het vervolg heeft.",
            "Het aantal woorden van een tekst is de lengte van de lijst met woorden ervan. En het aandeel is het aantal hits gedeeld door het aantal wat?",
            "`if same_start(written, truth) == ___:` en aan het eind `return hits / len(runs)`",
        ],
        why="`if same_start(written, truth) == len(truth.split()):` en aan het eind `return hits / len(runs)`. `share_extractable(RUNS)` geeft 0,714: 10 van de 14 runs.",
    )
    return


@app.cell(hide_code=True)
def text_after_share():
    mo.md(r"""
    Met 16 woorden context kwamen 10 van de 14 runs van Qwen3-0.6B-Base er helemaal uit: een aandeel van 0,714, en alle tien zijn het gekopieerde teksten. Van het chat model, met hetzelfde begin per tekst, kwam 1 van de 14 eruit: 0,071, lorem ipsum.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_chat():
    mo.md(r"""
    ### Oefening 4. Wat het chat model weglaat

    Met 16 woorden context schreef Qwen3-0.6B-Base 10 van de 12 gekopieerde teksten uit, en Qwen3-0.6B, het chat model, alleen lorem ipsum.
    """)
    return


@app.cell(hide_code=True)
def choice_chat_model_cell():
    choice_chat_model = L.choice(
        "chat-model",
        question="Wat zegt de uitkomst van het chat model?",
        options={
            "never": "Het chat model zag de andere elf teksten nooit",
            "right": "Wat deze prompt eruit haalt, niet wat het leerde",
            "safe": "Dat chat models veilig zijn, want ze geven niets van hun training data prijs",
        },
        correct="right",
        why={
            "right": "Volgens het rapport is elk Qwen3-model op dezelfde 36 biljoen tokens getraind, en de course gaf het chat model kale tekst, zonder het chat template waarin het leerde te antwoorden. Nasr en collega's zagen hetzelfde bij ChatGPT: gevraagd om teksten verder te schrijven waarvan bekend was dat het ze onthouden had, deed het dat in 3,5% van de gevallen, en hun poem-aanval haalde er toch meer dan 10.000 onthouden voorbeelden uit. Een test die niets vindt, laat niet zien dat er niets is.",
            "never": "Volgens het rapport is elk Qwen3-model op dezelfde 36 biljoen tokens getraind; pas daarna leerde het chat model te antwoorden. Dat deze prompt er niets uit haalt, betekent niet dat er niets in zit: Nasr en collega's haalden met een andere prompt onthouden tekst uit ChatGPT. Zie *Ook chat models* in *Het idee*.",
            "safe": "Nasr en collega's haalden voor 200 dollar meer dan 10.000 onthouden voorbeelden uit ChatGPT, en 16,9% van de gegenereerde teksten die ze testten, bevatte onthouden persoonsgegevens. Een chat model kan voor de ene prompt achterhouden wat het aan een andere prijsgeeft. Zie *Ook chat models* in *Het idee*.",
        },
    )
    choice_chat_model
    return (choice_chat_model,)


@app.cell(hide_code=True)
def text_exercise_carlini():
    mo.md(r"""
    ### Oefening 5. 604 van 1800

    Carlini en collega's (2021) bekeken 1800 teksten die GPT-2 schreef, uitgekozen omdat ze onthouden leken, en vonden daartussen 604 onthouden voorbeelden uit de training data.
    """)
    return


@app.cell(hide_code=True)
def choice_carlini_cell():
    choice_carlini = L.choice(
        "carlini",
        question="Wat zeggen deze getallen?",
        options={
            "personal": "604 teksten van GPT-2 bevatten namen, telefoonnummers of adressen van mensen",
            "right": "Minstens 604 onthouden teksten; 78 noemen mensen of geven contactgegevens",
            "third": "GPT-2 onthoudt een derde van alles waarop het getraind is, woord voor woord",
        },
        correct="right",
        why={
            "right": "604 van de 1800 kandidaten waren onthouden, volgens de publicatie 33,5%, en 67% in hun beste variant. Van alle 600.000 teksten die GPT-2 schreef, is dat minstens 0,1%, en dat noemen ze een ondergrens die waarschijnlijk ver onder het echte aantal ligt. De publicatie telt 78 voorbeelden van namen, telefoonnummers, adressen en accounts op sociale media van individuele mensen.",
            "personal": "604 is het aantal onthouden teksten van elke soort: nieuws, logbestanden, licenties, code en meer. De publicatie telt 78 voorbeelden van namen, telefoonnummers, adressen en accounts op sociale media van individuele mensen. Zie *Training data extraction* in *Het idee*.",
            "third": "33,5% is het aandeel van de 1800 kandidaten, uitgekozen omdat ze onthouden leken, niet van de training data. Van alle 600.000 teksten die GPT-2 schreef, bevatte minstens 0,1% onthouden tekst. Zie *Training data extraction* in *Het idee*.",
        },
    )
    choice_carlini
    return (choice_carlini,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Rekenen met geheimen](/course/open/rai-secret-sharing?lang=nl) verliet de data de eigenaren nooit. Een getraind model draagt echter stukken van zijn training data met zich mee, en hier heb je gemeten hoeveel: de meeste licenties en beroemde teksten, overal op het web gekopieerd, komen er woord voor woord uit als het begin lang genoeg is, en tekst die het model nooit zag niet. In [Wat is privacy?](/course/open/rai-what-is-privacy?lang=nl) zag je gepaste stroom: contactgegevens die een model voor iedereen uitschrijft, schenden die, ook als ze op het web stonden. In de volgende les, *Persoonsgegevens vinden en verbergen*, zoek je namen en telefoonnummers in tekst voordat die een model in gaat.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_carlini, choice_chat_model, result_same_start, result_share, result_split):
    L.wrap_up(result_split, result_same_start, result_share, choice_chat_model, choice_carlini)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
