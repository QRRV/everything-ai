# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-language-change. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-language-change", "nl")
    C = L.checks   # de woorden van de course, de plaatjes en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Taal, groepen en dialecten](/course/open/nlp-language-and-society?lang=nl) en één over [Corpora en treebanks](/course/open/nlp-corpora?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_variety():
    quick_variety = L.choice(
        "quick-variety",
        question="Een woordenlijst uit een treebank van standaardnederlands mist bijna de helft van de woorden van Groningse zinnen, en ongeveer 1 op de 16 woorden van de Nederlandse vertalingen. Wat zegt dat?",
        options={
            "right": "Een tool die op de lijst gebouwd is, kent minder woorden van het Gronings",
            "mistakes": "Mensen die Gronings schrijven, maken meer spelfouten dan mensen die Nederlands schrijven",
            "topics": "De Groningse zinnen gaan over andere dingen dan de vertalingen",
        },
        correct="right",
        why={
            "right": "De lijst kent de woorden van de variëteit waaruit de lijst gemaakt is. Deze les voegt een tweede reden toe waarom een lijst woorden kan missen: de tijd, want een taal verandert.",
            "mistakes": "Een onbekend woord zegt iets over de woordenlijst, niet over de spreker: *nait* is het Groningse woord voor *niet*, geen spelfout. Zie *Wat een tool kent* in [Taal, groepen en dialecten](/course/open/nlp-language-and-society?lang=nl).",
            "topics": "Elk paar zegt hetzelfde, dus het onderwerp is aan beide kanten gelijk. Zie *Wat een tool kent* in [Taal, groepen en dialecten](/course/open/nlp-language-and-society?lang=nl).",
        },
    )
    quick_variety
    return (quick_variety,)


@app.cell(hide_code=True)
def quick_check_concordance():
    quick_concordance = L.choice(
        "quick-concordance",
        question="Je wilt zien hoe een woord in een corpus gebruikt wordt. Wat geeft een concordantie van het woord je?",
        options={
            "counts": "Hoe vaak elk woord van het corpus voorkomt, van het meest voorkomende woord naar beneden",
            "right": "Elke plek waar het woord staat, met de woorden eromheen",
            "trees": "De boom van elke zin waarin het woord staat",
        },
        correct="right",
        why={
            "right": "Een concordantie zet elke plek waar het woord staat onder elkaar, met de buren aan beide kanten. Deze les gebruikt er een om te vergelijken hoe een woord in romans uit de 19e eeuw gebruikt werd en hoe het nu gebruikt wordt.",
            "counts": "Dat is een woordenlijst met aantallen. Wat zet een concordantie onder elkaar? Zie *Elke plek van een woord: de concordantie* in [Corpora en treebanks](/course/open/nlp-corpora?lang=nl).",
            "trees": "Dat is wat een treebank aan de zinnen toevoegt. Wat zet een concordantie onder elkaar? Zie *Elke plek van een woord: de concordantie* in [Corpora en treebanks](/course/open/nlp-corpora?lang=nl).",
        },
    )
    quick_concordance
    return (quick_concordance,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Elke tekst is door iemand geschreven, ergens, op een bepaald moment. Jurafsky en Martin schrijven in het hoofdstuk *Words and Tokens* van *Speech and Language Processing* dat ook de tijd telt: een taal verandert in de loop van de tijd, en voor sommige talen bestaan goede corpora met teksten uit verschillende periodes. Een model leert de taal van de data, en die taal verandert verder. In 2021 trainden Angeliki Lazaridou en anderen modellen op nieuws en wetenschappelijke samenvattingen tot september 2017, en testten de modellen op tekst uit 2018 en 2019: bij teksten die tot twee jaar na het einde van de trainingsperiode verschenen, deden de modellen het tot 16% slechter. In een van hun verzamelingen nieuws was ongeveer 27% van de verschillende woorden van elke maand nooit voorgekomen in de trainingsperiode, zoals *Brexiteers* en *MeToo*. Talen ontmoeten elkaar ook. Mensen die twee talen kennen, gebruiken vaak beide talen, zelfs in één zin: volgens Jurafsky en Martin, in hetzelfde hoofdstuk, komt code-switching overal ter wereld enorm veel voor, en een tool die één taal per tekst verwacht, kan over zulke zinnen struikelen. Deze les kijkt naar zinnen die van taal wisselen, en telt hoe het Nederlands veranderde tussen drie romans uit de 19e eeuw en zinnen van nu.

    **Een voorbeeld uit het dagelijks leven.** Oude brieven, recepten en kranten van honderd jaar geleden schrijven *menschen*, *tusschen* en *oogen*: woorden die je kent, in een spelling die je zelf niet zou gebruiken.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In drie Nederlandse romans uit de 19e eeuw, samen 469.251 woorden, staat het woord *zoo* 2.587 keer. Hoe vaak komt *zoo* voor in de 1.301.039 woorden van de Nederlandse zinnen van Tatoeba, geschreven door vrijwilligers van nu?",
        options={
            "zero": "0 keer",
            "right": "15 keer",
            "few": "Ongeveer 300 keer",
            "same": "Ongeveer 2.500 keer",
        },
        reveal="**15 keer**, en elke keer is het een ander woord: *zoo* voor een dierentuin, zoals in het Engels, in zinnen als *Ik ben naar de zoo geweest*. De *zoo* van de romans is de oude spelling van *zo*, dat de romans maar 3 keer schrijven, en de zinnen van nu 3.154 keer. Eén vorm, twee woorden: een spelling die veranderde, en een woord voor een plek met dieren.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Lenen

    Talen nemen woorden van elkaar over. De World Loanword Database (WOLD), onder redactie van Martin Haspelmath en Uri Tadmor, noemt een woord een leenwoord als het ergens in de geschiedenis van een taal uit een andere taal gekopieerd is. Het Engels heeft *cookie* uit het Nederlandse *koekje*, *coleslaw* uit *koolsla*, *yacht* uit een ouder Nederlands *jaght*, en *boss* uit *baas*, volgens het woordenboek van Merriam-Webster. Het Nederlands nam *trottoir* en *cadeau* over uit het Frans, en *tram*, *weekend* en *computer* uit het Engels. In de woordenlijst van WOLD voor het Nederlands, 1.513 woorden voor een vaste lijst alledaagse betekenissen, is 19% duidelijk of waarschijnlijk geleend; in de lijst voor het Engels 42%. Hoeveel van een taal geleend is, hangt af van wat je telt. In een telling van Nicoline van der Sijs van vier pagina's van de krant NRC Handelsblad uit 1994 was 30,7% van de 2.144 verschillende woorden ooit uit een andere taal in het Nederlands gekomen, en geteld op elke plek waar een woord voorkwam, zakte het aandeel naar 16,2%. Lenen gaat twee kanten op: het boek *Nederlandse woorden wereldwijd* van Van der Sijs, uit 2010, noemt 17.560 Nederlandse woorden die in een of meer van 138 talen terechtkwamen.

    ### Code-switching

    Mensen die twee talen kennen, gebruiken vaak beide talen in één gesprek, of in één zin: **code-switching**. Shana Poplack omschreef het in 1980 als het afwisselen van twee talen binnen één gesprek, één zin of één zinsdeel. De titel van die publicatie is een zin van een spreker uit het onderzoek van Poplack: *Sometimes I'll start a sentence in Spanish y termino en español*, soms begin ik een zin in het Spaans en eindig ik in het Spaans. Poplack onderscheidde drie soorten: een wissel tussen zinnen, een los ingevoegd stukje zoals een tussenwerpsel of een vaste uitdrukking, door Poplack een tag genoemd, en een wissel binnen een zin, die volgens Poplack de meeste vaardigheid vraagt. Ad Backus onderzocht hoe sprekers van het Turks in Nederland Turks en Nederlands samen gebruiken, in zinnen als *Babama bir smoes verzinnen yaptik, schoolfeestje var diye*: voor mijn vader hebben we een smoes verzonnen, we zeiden dat er een schoolfeestje was. Een wissel kan zelfs midden in een woord vallen: in *nieuwkomers'a* krijgt het Nederlandse woord *nieuwkomers* de Turkse uitgang *-a*, waar het Nederlands *naar* heeft.

    Lenen en code-switching zijn niet altijd makkelijk uit elkaar te houden. A. Seza Doğruöz en anderen schrijven in een overzicht uit 2021 dat code-switching een actieve afwisseling tussen twee grammatica's inhoudt, en lenen niet. Bij één los woord uit een andere taal is moeilijk te zeggen of het een leenwoord is of een wissel, tenzij je kijkt hoe het woord in de grammatica van de andere taal past, schrijven ze. Voor tools is code-switching lastig: volgens het overzicht kan taaltechnologie code-switching nog niet zo soepel verwerken als tekst in één taal. In 2014 organiseerden Thamar Solorio en anderen een shared task, een wedstrijd waarin onderzoeksteams dezelfde taak op dezelfde data uitvoeren: de teams moesten elk woord van teksten met code-switching een van zes labels geven, zoals de eerste taal, de tweede taal en gemengd. Zeven teams deden mee. Bepalen in welke taal een tekst of een woord is, heet language identification, en volgens de organisatoren was language identification in tekst met code-switching nog lang niet opgelost, en was er meer onderzoek nodig.

    ### Betekenisverandering

    Een woord kan ook voor iets anders gaan staan, terwijl de vorm blijft: **betekenisverandering**. Een woord kan minder gaan betekenen: het Oudengelse *dēor* kon voor veel soorten dieren staan, en werd het Engelse *deer*, hert. Het kan ook iets heel anders gaan betekenen: het Engelse *silly* gaat terug op het Oudengelse *sǣlig*, gelukkig of gezegend, *nice* betekende ooit dwaas, en *awful* ooit vol ontzag. Het Nederlands kent hetzelfde. Volgens het *Etymologisch Woordenboek van het Nederlands* betekende *slim* eerst scheef, toen slecht, en vanaf 1602 is het te vinden in de betekenis schrander. *Gemeen* betekende gemeenschappelijk, zoals nog in *iets gemeen hebben*, toen gewoon, en pas vanaf de 18e eeuw iets negatiefs: laag. *Erg* betekende slecht; het woordenboek vindt het pas in 1867 in de betekenis *heel*. *Zeer*, dat pijnlijk betekende, zoals nog in *zeer doen*, was in het Middelnederlands al een woord voor *heel*. En *schoon* en de verwante woorden betekenden in alle Germaanse talen eerst mooi; in Nederland is het in die betekenis nu ouderwets, behalve in vaste combinaties zoals *de schone kunsten*, en betekent het rein, niet vuil, terwijl in België de betekenis mooi nog gewoon is, zoals in het Duits.

    Embeddings, die een latere les van dit domein bouwt in *Word embeddings: buren en analogieën*, kunnen zulke veranderingen meten. William Hamilton, Jure Leskovec en Dan Jurafsky trainden embeddings op de teksten van elk decennium van zes historische corpora in vier talen, en vonden twee wetten. Volgens de eerste veranderen veelvoorkomende woorden langzamer van betekenis. Volgens de tweede veranderen woorden met veel betekenissen sneller, ook vergeleken met woorden die even vaak voorkomen.

    ### Spelling

    Ook de spelling verandert, vaak door een besluit. Het Nederlands kreeg in 1804 de eerste officiële spelling, van Siegenbeek; de woordenlijst van De Vries en Te Winkel volgde rond 1865. In 1934 vereenvoudigde minister Marchant de spelling voor scholen, en een wet uit 1947 in Nederland, na een besluit uit 1946 in België, maakte een spelling met die veranderingen officieel. Het eerste *Groene Boekje*, de officiële woordenlijst, verscheen in 1954, en nieuwe spellingregels kwamen er in 1995 en 2005. De memorie van toelichting uit 2005 bij de Spellingwet vat samen wat er in 1947 veranderde: een dubbele *e* en *o* aan het eind van een lettergreep werden enkel, *sch* werd *s* waar je de *ch* niet uitspreekt, en de naamvalsuitgangen werden niet meer verplicht. Zo werd *oogen* voortaan *ogen* en *menschen* voortaan *mensen*; en *den* en *eene*, naamvalsvormen van *de* en *een*, zijn zeldzaam in de zinnen van Tatoeba.

    De drie Nederlandse romans van de course verschenen voor het eerst in 1839, 1860 en 1889, en de bestanden van de course bevatten latere drukken: *Camera Obscura* is de 29e druk, uit 1917, en *Max Havelaar* een uitgave uit 1881 of later, die *y* schrijft voor *ij*, zoals in *hy* en *zyn*. Alle drie staan in spelling van vóór 1934. Ook de Engelse spelling verandert: de meeste Engelse romans van de course schrijven *to-day* en *to-morrow*.

    ### Elk corpus hoort bij een tijd

    Een corpus laat dus de taal van een tijd en van bepaalde mensen zien. De drie romans bevatten het Nederlands van drie schrijvers uit de 19e eeuw, in drukken van vóór 1934; de zinnen van Tatoeba het Nederlands van vrijwilligers van nu, korte zinnen van gemiddeld 6,5 woorden. Als je de twee vergelijkt, komt een deel van het verschil door de tijd, en een deel door wie wat schreef, en voor wie. Jurafsky en Martin raden in hetzelfde hoofdstuk aan om elk corpus te beschrijven met vragen als: wanneer en in welke situatie is de tekst geschreven of gesproken, en in welke taal, met welk dialect of welke regio? Het plaatje laat een zin zien die van taal wisselt, en een zin uit een van de romans in de oude spelling.
    """)
    return


@app.cell(hide_code=True)
def idea_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **leenwoord** | een woord dat een taal uit een andere taal heeft overgenomen, zoals het Engelse *cookie* uit het Nederlandse *koekje* |
        | **code-switching** | twee talen gebruiken in één gesprek of één zin: tussen zinnen, als los ingevoegd stukje, of binnen een zin |
        | **language identification** | bepalen in welke taal een tekst of een woord is |
        | **betekenisverandering** | een verandering in waar een woord voor staat, terwijl de vorm blijft, zoals bij *slim*: scheef, toen slecht, toen schrander |
        | **officiële spelling** | de manier van schrijven die een overheid voor een taal vastlegt, zoals de spelling van het *Groene Boekje* |
        | **per 10.000 woorden** | hoe vaak een woord voorkomt op elke 10.000 woorden van een corpus: het aantal gedeeld door alle woorden van het corpus, maal 10.000, zodat je corpora van verschillende grootte kunt vergelijken |
        | **taaltag** | een korte code voor de taal van een woord, zoals `"nl"` voor Nederlands en `"tr"` voor Turks |
        """),
        L.watch_out("Een verschil tussen twee corpora komt niet alleen door verandering in de loop van de tijd: de romans en de zinnen van Tatoeba verschillen ook in schrijvers, onderwerpen en lezers. En een oude spelling is geen fout: *menschen* was de goede spelling toen deze romans gedrukt werden."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Om code-switching te vinden, geef je elk woord van een zin een taaltag, en loop je langs de zin terwijl je elke tag vergelijkt met de tag ervoor.
    2. Om een woord tussen toen en nu te vergelijken, tel je het in een corpus uit elke tijd, en deel je elk aantal door de grootte van het eigen corpus, en vermenigvuldig je met 10.000: per 10.000 woorden.
    3. Om te zien hoe een woord gebruikt wordt, tel je in elk corpus welke woorden erna komen, en lees je een paar plekken waar het staat.
    4. Lees de getallen samen met wat de corpora zijn: wie de teksten schreef, wanneer en voor wie.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De eerste cel bevat de titelzin van de publicatie van Poplack als lijst woorden, `POPLACK`, en de taal van elk woord als lijst tags, `POPLACK_TAGS`: `"en"` voor Engels en `"es"` voor Spaans. De tweede cel telt hoeveel woorden elke taal heeft, in een dictionary van de tag naar het aantal.
    """)
    return


@app.cell
def poplack_cell():
    POPLACK = C.POPLACK
    POPLACK_TAGS = C.POPLACK_TAGS
    print(POPLACK)
    print(POPLACK_TAGS)
    return (POPLACK_TAGS,)


@app.cell
def example_per_language(POPLACK_TAGS):
    def per_language(tags):
        counts = {}
        for tag in tags:
            if tag in counts:
                counts[tag] = counts[tag] + 1
            else:
                counts[tag] = 1
        return counts

    print(per_language(POPLACK_TAGS))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De zin heeft zeven Engelse woorden en vier Spaanse. De aantallen zeggen welke talen een zin heeft, maar niet waar de zin wisselt: de aantallen zouden hetzelfde blijven als je de woorden door elkaar husselt. Om de wissels te vinden, kijk je naar elk woord samen met het woord ervoor.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_switches():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. switches: waar een zin van taal wisselt

        `BACKUS` hieronder bevat de zin van Ad Backus uit het plaatje als lijst woorden, en `BACKUS_TAGS` de taal van elk woord: `"tr"` voor Turks en `"nl"` voor Nederlands. Schrijf `switches(tags)`: het aantal plekken waar een woord een andere tag heeft dan het woord ervoor. Voor `switches(BACKUS_TAGS)` is de uitkomst 4: van Turks naar Nederlands, terug naar Turks, weer naar Nederlands, en terug naar Turks. Een tag `"mixed"`, voor een woord dat half de ene taal is en half de andere, telt als een eigen tag. Vervang de `...`. De check roept `switches(BACKUS_TAGS)` aan, en daarna eigen gevallen: de titel van Poplack, een zin met een gemengd woord, één woord, geen woorden, een wissel bij elk woord, en een zin in één taal.
        """),
        L.task("Tel waar een zin van de ene taal naar de andere wisselt."),
    ])
    return


@app.cell
def backus_cell():
    BACKUS = C.BACKUS
    BACKUS_TAGS = C.BACKUS_TAGS
    print(BACKUS)
    print(BACKUS_TAGS)
    return


@app.cell(hide_code=True)
def exercise_switches():
    box_switches = L.code_box("switches", lines=6)
    box_switches
    return (box_switches,)


@app.cell(hide_code=True)
def check_switches(box_switches):
    result_switches = L.check("switches", box_switches.value)
    result_switches
    return (result_switches,)


@app.cell(hide_code=True)
def hints_switches():
    L.hints(
        "switches",
        [
            "Begin met een teller `n` op 0. Loop langs de plekken van de lijst, van de tweede plek tot het eind, en vergelijk de tag op elke plek met de tag op de plek ervoor. Als de twee tags verschillen, tel je 1 op bij `n`. Geef na de loop `n` terug.",
            "Je staat op plek `i`. Op welke plek staat de tag van het woord ervoor?",
            "`        if tags[i] != tags[___]:`",
        ],
        why="`n = 0`, dan `for i in range(1, len(tags)):` met daaronder `if tags[i] != tags[i - 1]:` en `n = n + 1`. Na de loop: `return n`.",
    )
    return


@app.cell(hide_code=True)
def text_after_switches():
    mo.md(r"""
    De titel van Poplack wisselt één keer, van Engels naar Spaans. De eerste zin van Backus wisselt vier keer, en de tweede, *Geldiğimde iki ay nieuwkomers'a gittim*, gaat van Turks naar een gemengd woord en terug: een tag per woord vangt een wissel binnen een zin, en de tag gemengd een wissel binnen een woord. Een tool die een hele tekst één taal geeft, kan niet laten zien waar een zin wisselt; in de wedstrijd van Solorio en anderen kreeg elk woord een eigen label.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_per():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. per_10k: corpora van verschillende grootte vergelijken

        `NOVEL_COUNTS` hieronder bevat hoe vaak elk woord voorkomt in de drie romans, in kleine letters, en `TATOEBA_COUNTS` hetzelfde voor de Nederlandse zinnen van Tatoeba. De romans hebben 469.251 woorden, de zinnen 1.301.039: met kale aantallen lijkt een woord in het grotere corpus al snel vaker voor te komen, alleen omdat dat corpus groter is. Schrijf `per_10k(counts, word)`: hoe vaak `word` voorkomt op elke 10.000 woorden van het corpus, dus het aantal gedeeld door het aantal van alle woorden van het corpus, de som van alle aantallen, maal 10.000. Een woord dat niet in `counts` staat, komt 0 keer voor. Voor `per_10k(NOVEL_COUNTS, "zoo")` is de uitkomst 55,13. Vervang de `...`. De check roept `per_10k(NOVEL_COUNTS, "zoo")` aan, en daarna eigen gevallen: *zoo*, *zo* en *menschen* in Tatoeba, een kleine dictionary met aantallen, een woord dat er niet in staat, en een lege dictionary.
        """),
        L.task("Bereken hoe vaak een woord voorkomt per 10.000 woorden, om corpora van verschillende grootte te vergelijken."),
    ])
    return


@app.cell
def counts_cell():
    NOVEL_COUNTS = C.novel_counts()
    TATOEBA_COUNTS = C.tatoeba_counts()
    print(len(NOVEL_COUNTS), NOVEL_COUNTS["zoo"], len(TATOEBA_COUNTS), TATOEBA_COUNTS["zo"])
    return


@app.cell(hide_code=True)
def exercise_per():
    box_per = L.code_box("per-10k", lines=7)
    box_per
    return (box_per,)


@app.cell(hide_code=True)
def check_per(box_per):
    result_per = L.check("per-10k", box_per.value)
    result_per
    return (result_per,)


@app.cell(hide_code=True)
def hints_per():
    L.hints(
        "per-10k",
        [
            "Tel alle aantallen op in een variabele `total`: loop langs de woorden van `counts` en tel bij elk woord het aantal op. Geef daarna 0 terug als `word` niet in `counts` staat. Deel anders het aantal van `word` door `total`, en schaal de uitkomst op.",
            "In de loop is `w` één woord van `counts`. Hoe kom je aan hoe vaak dat woord voorkomt?",
            "`        total = total + ___`",
        ],
        why="`total = 0`, dan `for w in counts:` met daaronder `total = total + counts[w]`. Na de loop: `if word not in counts:` met `return 0`, en dan `return counts[word] / total * 10000`.",
    )
    return


@app.cell(hide_code=True)
def text_after_per():
    mo.md(r"""
    Elke oude spelling komt veel voor in de romans en weinig of niet in de zinnen van nu, en bij elke nieuwe spelling is het andersom: *oogen* komt in de romans 10,7 keer per 10.000 woorden voor en in Tatoeba geen enkele keer, *ogen* in de romans geen enkele keer en in Tatoeba 3,9 keer. Dat is de verandering van 1934 en 1947 in getallen. Per 10.000 woorden gebruiken de romans *zoo* ook vaker dan Tatoeba *zo*, 55,1 tegen 24,2 keer: dat zijn andere teksten, met een andere stijl, en het getal zegt niet waarom. Zonder te delen zou *nodig* (1.959 keer in Tatoeba) 11 keer zo vaak lijken voor te komen als *noodig* (176 keer in de romans); per 10.000 woorden is het 4 keer zo vaak.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_next():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. next_words: in welk gezelschap een woord staat

        `NOVELS` hieronder bevat de zinnen van de drie romans, geknipt bij elke punt, elk vraagteken en elk uitroepteken, elk als lijst van de woorden; `TATOEBA` bevat elke Nederlandse zin van Tatoeba, van tatoeba.org, als lijst van de woorden. Schrijf `next_words(sentences, target)`: tel welke woorden direct na `target` komen, in een dictionary van het volgende woord, in kleine letters, naar hoe vaak het volgt. Vergelijk elk woord in kleine letters met `target`, zodat *Schoon* aan het begin van een zin ook telt; `target` zelf krijg je in kleine letters. Het laatste woord van een zin heeft geen volgend woord. Voor `next_words(TATOEBA, "schoon")` heeft de dictionary 23 woorden, en volgt `"te"` 20 keer. Vervang de `...`. De check roept `next_words(TATOEBA, "schoon")` aan, en daarna eigen gevallen: *schoon* en *erg* in de romans, het woord met een hoofdletter en aan het eind van een zin, een volgend woord met een hoofdletter, het woord twee keer in één zin, een zin van één woord, en geen zinnen.
        """),
        L.task("Tel welke woorden na een woord komen in een corpus, om te zien hoe het gebruik veranderde."),
    ])
    return


@app.cell
def sentences_cell():
    NOVELS = C.novels()
    TATOEBA = C.tatoeba()
    print(len(NOVELS), NOVELS[100])
    print(len(TATOEBA), TATOEBA[100])
    return


@app.cell(hide_code=True)
def exercise_next():
    box_next = L.code_box("next-words", lines=11)
    box_next
    return (box_next,)


@app.cell(hide_code=True)
def check_next(box_next):
    result_next = L.check("next-words", box_next.value)
    result_next
    return (result_next,)


@app.cell(hide_code=True)
def hints_next():
    L.hints(
        "next-words",
        [
            "Begin met een lege dictionary `counts`. Loop langs de zinnen, en daarbinnen langs de plekken van elke zin, tot de plek vóór het laatste woord. Als het woord op een plek in kleine letters gelijk is aan `target`, neem je het woord op de volgende plek, in kleine letters, en werk je het aantal ervan bij. Geef na beide loops de dictionary terug.",
            "Het woord `target` staat op plek `i`. Op welke plek staat het woord dat erna komt?",
            "`                w = sentence[___].lower()`",
        ],
        why="`counts = {}`, dan `for sentence in sentences:` en daarbinnen `for i in range(len(sentence) - 1):`, met `if sentence[i].lower() == target:` en daaronder `w = sentence[i + 1].lower()`, dan `if w in counts:` met `counts[w] = counts[w] + 1`, en `else:` met `counts[w] = 1`. Na beide loops: `return counts`.",
    )
    return


@app.cell(hide_code=True)
def text_after_next():
    mo.md(r"""
    In de romans komen na *schoon* het vaakst *ik*, *en* en *was*. Met *ik* erna betekent het hoewel, zoals in *schoon ik gaarne beken*, al geef ik graag toe, een gebruik dat volgens het etymologisch woordenboek van De Vries teruggaat op een betekenis die *schoon* tegen 1500 kreeg; elders betekent het mooi, zoals in *hoe schoon en voortreffelijk*. In Tatoeba komt na *schoon* het vaakst *te*, zoals in *schoon te maken*, en verder *houden* en *drinkwater*: daar betekent het rein. *Erg* wordt in de romans 41 keer door een woord gevolgd, het vaakst door *best* en *boos*, en in Tatoeba 1.824 keer, het vaakst door *goed*, *van* en *druk*. Per 10.000 woorden ging *erg* van 1,0 in de romans naar 14,3 in Tatoeba, en *zeer* van 18,2 naar 3,6: in de romans is *zeer* het gewone woord om iets sterker te maken, zoals in *zeer goed*; in Tatoeba zijn dat *erg* en *heel*. *Erg* ging van een woord voor iets slechts naar een woord voor *heel*, net als *zeer* (eerst pijnlijk) eeuwen eerder.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: een woord toen en nu

    Typ een woord. Het plaatje laat zien hoe vaak het per 10.000 woorden voorkomt in de drie romans en in de zinnen van Tatoeba, en een paar plekken waar het in elk corpus staat, met de woorden eromheen: een concordantie. Probeer *schoon*, *erg*, *zeer*, *gaarne*, *mevrouw*, *leuk* of een eigen woord.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_word = mo.ui.text(value="schoon", label="Woord")
    demo_word
    return (demo_word,)


@app.cell(hide_code=True)
def demo_look(demo_word):
    mo.Html(C.demo_picture(demo_word.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    *Gaarne* komt 165 keer voor in de romans en geen enkele keer in Tatoeba, waar *graag* 1.505 keer voorkomt. *Mevrouw* gaat van 18,0 naar 0,5 per 10.000 woorden, en *leuk* van geen enkele keer in de romans naar 6,9 in Tatoeba. Een deel daarvan is verandering, en een deel is het verschil tussen romans uit de 19e eeuw en korte alledaagse zinnen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Oefening 4. Geleend, gewisseld of veranderd?

    Elke vetgedrukte uitspraak laat één manier zien waarop talen elkaar ontmoeten of veranderen. Zet elke uitspraak bij de manier die erbij hoort.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_change = L.sort("borrow-switch-change")
    sort_change
    return (sort_change,)


@app.cell(hide_code=True)
def check_sort(sort_change):
    result_sort = L.check("borrow-switch-change", sort_change)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Taal, groepen en dialecten](/course/open/nlp-language-and-society?lang=nl) liet zien dat een taal verschilt tussen regio's, groepen en situaties; deze les voegde tijd en contact toe. Talen lenen woorden van elkaar, en hoeveel van een taal geleend is, hangt af van wat je telt. Mensen die twee talen kennen, wisselen er vaak tussen, zelfs binnen een zin, en tools kunnen daar nog minder goed mee overweg dan met tekst in één taal. Woorden gaan voor iets anders staan, zoals *slim* en *schoon*, en de spelling verandert, vaak door een besluit, zoals in 1934 en 1947. Je telde waar zinnen van taal wisselen, vergeleek oude en nieuwe spellingen per 10.000 woorden in twee corpora van verschillende grootte, en telde de woorden na *schoon* en *erg* toen en nu. De concordantie van [Corpora en treebanks](/course/open/nlp-corpora?lang=nl) liet je elke plek van een woord zien; als je de buren over een heel corpus telt, zie je het gebruik in getallen. Een corpus laat de taal van een tijd en van bepaalde mensen zien, dus beschrijf wanneer, door wie en voor wie het geschreven is. Deze les sluit de module over taalkunde af; de volgende les van het domein begint met het verwerken van tekst: *Patronen vinden: reguliere expressies*.

    **Een voorbeeld uit het dagelijks leven.** Als een vertaalapp een ouderwets woord geeft, of een nieuw woord niet kent, vraag je dan af uit welke jaren de data kwam.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_next, result_per, result_sort, result_switches):
    L.wrap_up(result_switches, result_per, result_next, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
