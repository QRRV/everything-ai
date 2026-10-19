# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les llm-dedup. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-dedup", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Tekst opschonen: quality filters](/course/open/llm-quality-filter?lang=nl) en [Tuples, sets en sorteren](/course/open/py-tuples-sets?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rules():
    quick_rules = L.choice(
        "quick-rules",
        question="Een pagina vraagt je te bewijzen dat je een mens bent, in vier lange, nette zinnen. Houden de regelfilters van C4 hem?",
        options={
            "right": "Ja: zijn regels sluiten een zin af en hebben minstens 5 woorden",
            "no": "Nee: C4 weet dat zulke meldingen geen echte tekst zijn, en haalt ze weg",
            "page": "Nee: een pagina over mensen gaat eruit",
        },
        correct="right",
        why={
            "no": "De regelfilters kijken alleen hoe een regel eindigt, hoeveel woorden hij heeft en of hij JavaScript noemt: niet of de tekst het lezen waard is. Zie [Tekst opschonen: quality filters](/course/open/llm-quality-filter?lang=nl).",
            "page": "Geen filter van C4 of Gopher vraagt waar een pagina over gaat: ze kijken naar regels, woorden en leestekens, en C4 naar een lijst van slechte woorden. Zie [Tekst opschonen: quality filters](/course/open/llm-quality-filter?lang=nl).",
            "right": "Regels over de vorm van tekst laten zo'n melding door, op elke site die hem toont. Deze les vindt de kopieën.",
        },
    )
    quick_rules
    return (quick_rules,)


@app.cell(hide_code=True)
def quick_check_set():
    quick_set = L.choice(
        "quick-set",
        question="Wat laat dit programma zien?\n\n```python\nshingles = {\"the cat\", \"cat sat\"}\nprint(shingles[0])\n```",
        options={
            "first": "`the cat`: de waarde die als eerste staat",
            "sorted": "`cat sat`: een set bewaart zijn waarden van A tot Z",
            "right": "Een fout: een set heeft geen plekken, dus ook geen index",
        },
        correct="right",
        why={
            "first": "Een set heeft geen plekken, dus ook geen eerste waarde: `shingles[0]` stopt met een fout. Zie [Tuples, sets en sorteren](/course/open/py-tuples-sets?lang=nl).",
            "sorted": "Een set heeft geen volgorde, ook niet van A tot Z, dus geen plek 0: `shingles[0]` stopt met een fout. Zie [Tuples, sets en sorteren](/course/open/py-tuples-sets?lang=nl).",
            "right": "Een set weet alleen welke waarden erin zitten, elk één keer. Meer heeft deze les niet nodig: welke reeksen woorden twee pagina's delen, niet waar.",
        },
    )
    quick_set
    return (quick_set,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Veel webpagina's zijn bijna gelijk aan een andere: dezelfde melding op elke site die dezelfde dienst gebruikt, hetzelfde artikel met een andere datum of een ander menu, dezelfde tekst gekopieerd naar honderd webwinkels. Een model dat een tekst vaak ziet, leert hem uit het hoofd. In C4 stond één zin van 61 woorden er meer dan 60.000 keer in, en modellen die op zulke corpora getraind waren, namen meer dan 1% van wat ze zonder prompt schreven woord voor woord over uit hun training data. Na deduplicatie, van exacte kopieën en near-duplicates, deden ze dat tien keer minder vaak, en hadden ze minder stappen nodig om net zo goed te leren (Lee en anderen, 2022). Maar elke pagina met elke andere vergelijken is veel te traag voor miljoenen pagina's. Deze les laat een truc zien die de near-duplicates vindt zonder dat.

    **Een voorbeeld uit het dagelijks leven.** Twee foto's van dezelfde straat, een minuut na elkaar genomen, zijn niet hetzelfde bestand, maar je ziet meteen dat het bijna dezelfde foto is. Om zulke paren te vinden tussen 10.000 foto's, wil je liever niet elke foto naast elke andere houden.
    """)
    return


@app.cell(hide_code=True)
def predict_copies():
    _m = C.MEASURED
    guess_copies = L.guess(
        "copies",
        question=f"Van de {C.number(C.Q.MEASURED['pages'], 0, L.lang)} ruwe pagina's van de download van [Tekst opschonen: quality filters](/course/open/llm-quality-filter?lang=nl) komen er {C.number(_m['pages'], 0, L.lang)} door al zijn filters. Hoeveel daarvan zijn volgens jou een near-duplicate van een andere pagina, en kunnen weg als er van elk groepje één blijft?",
        options={
            "none": "Geen of bijna geen",
            "right": "Ongeveer 1 op de 75",
            "fifth": "Ongeveer 1 op de 5",
        },
        reveal=f"**Ongeveer 1 op de 75: {_m['removed']} pagina's, in {_m['groups']} groepjes.** Het grootste groepje is {_m['largest']} kopieën van een melding die je vraagt een CAPTCHA op te lossen; andere zijn de melding boven een XML-sitemap, en een forummelding dat je moet inloggen. Slechts {_m['exact_extra']} van de {_m['removed']} zijn exacte kopieën; de andere verschillen in een titel, een naam, een telefoonnummer of een datum. In een hele crawl, met miljoenen pagina's van dezelfde sites, zijn het er veel meer.",
    )
    guess_copies
    return (guess_copies,)


@app.cell(hide_code=True)
def text_idea():
    _agree, _j = C.example_agree(24)
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Exacte kopieën zijn makkelijk.** Stop elke tekst in een set of een dict: een tekst die er al in zit, is een kopie.

        **Near-duplicates vragen om een maat.** Knip elke pagina in shingles: al zijn reeksen van $k$ woorden achter elkaar, hier 5. Twee pagina's die in één woord verschillen, delen nog bijna al hun shingles. Hun overlap, de **Jaccard similarity**, is het aantal shingles dat ze delen gedeeld door het aantal verschillende shingles van de twee samen: 1 voor dezelfde pagina, 0 voor pagina's zonder een gezamenlijke reeks van 5 woorden. Twee pagina's die minstens 0,8 delen, gelden als near-duplicates.

        **Alle paren zijn te veel.** 5.614 pagina's geven 15.755.691 paren; een miljard pagina's geven er ongeveer $5 \cdot 10^{17}$. Daarom krijgt elke pagina in plaats daarvan een korte signature, met MinHash:

        1. Geef elke shingle een getal met een hashfunctie: een vaste functie van tekst naar een getal, dezelfde tekst altijd naar hetzelfde getal. Hier CRC-32, uit zlib van Python, genomen `% P` zodat het onder $P = 2^{31} - 1$ ligt.
        2. Maak er 128 hashfuncties bij: functie $i$ maakt van een hash $h$ het getal $(a_i h + b_i) \bmod P$, met een eigen willekeurige $a_i$ en $b_i$. Elke functie zet de shingles in een andere volgorde, alsof ze geschud zijn.
        3. Houd voor elke functie het kleinste getal dat hij aan een shingle van de pagina geeft: die 128 getallen zijn de signature van de pagina.

        **Waarom het werkt.** Neem één functie, en alle shingles van twee pagina's samen. Elk ervan heeft ongeveer evenveel kans om het kleinste getal te krijgen. De twee signatures zijn op die plek precies dan gelijk als die shingle in beide pagina's zit. Dus de kans dat een plek gelijk is, is ongeveer hun overlap, en het deel van de 128 plekken dat gelijk is, schat die: bij de 594 paren die de demo vergelijkt, gemiddeld tot op 0,012.

        **Banding vindt de paren.** Knip elke signature in 16 banden van 8 getallen. Twee pagina's die in alle 8 getallen van een band gelijk zijn, komen in dezelfde bucket, en alleen pagina's die een bucket delen, worden vergeleken. Een paar met een overlap van 0,9 deelt een band met een kans van 99,99%, een paar van 0,8 met een kans van 95%, en een paar van 0,3 met een kans van 0,1%. Dit is locality-sensitive hashing: hier 594 paren om te vergelijken in plaats van 15,8 miljoen.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **deduplicatie** | | kopieën uit een dataset verwijderen, exacte kopieën en near-duplicates, zodat een model niet vaak dezelfde tekst ziet | |
        | **near-duplicate** | | een document dat bijna gelijk is aan een ander, zoals dezelfde melding met een andere naam; zoeken naar exacte kopieën vindt het niet | |
        | **shingling** | | een document knippen in al zijn overlappende reeksen van $k$ woorden, zijn shingles | `shingles(text, 5)` |
        | **Jaccard similarity** | | de overlap van twee sets: de items die ze delen gedeeld door al hun verschillende items samen; 1 voor dezelfde set, 0 voor sets zonder een gedeeld item | `len(a & b) / len(a \| b)` |
        | **MinHash** | | een signature van een document: voor elk van veel hashfuncties het kleinste getal dat hij aan een shingle geeft; twee signatures zijn op ongeveer zoveel plekken gelijk als de documenten overlappen | `minhash(hashes, a, b)` |
        | **locality-sensitive hashing** | | items die op elkaar lijken met grote kans in dezelfde bucket stoppen, zodat je alleen items in dezelfde bucket hoeft te vergelijken; hier: banding van de signatures | |
        | `&`, `\|`, `-` op sets | | `a & b` is de set van de items in beide sets, `a \| b` van de items in minstens één, `a - b` van de items in `a` maar niet in `b` | `{1, 2} & {2, 3}` geeft `{2}` |
        | **outer product** | | een tabel van elk getal van de ene array keer elk getal van de andere: rij `r`, kolom `c` bevat `h[r] * a[c]` | `np.outer(h, a)` |
        | $\bmod$ | modulo | de rest na delen: $10 \bmod 3 = 1$ | `10 % 3` |
        """),
        mo.md(r"""
        Het beeld: de signatures van de twee teksten van het uitgewerkte voorbeeld, met 24 hashfuncties, plek voor plek. Een gevuld vakje is een plek waar de twee signatures hetzelfde getal hebben.
        """),
        mo.Html(C.signatures_picture(_agree, {"title": "Waar twee signatures gelijk zijn", "row": f"de {len(_agree)} plekken van de signatures van tekst 1 en tekst 2",
                                              "notes": [f"{sum(_agree)} van de {len(_agree)} plekken zijn gelijk: een schatting van {C.number(sum(_agree) / len(_agree), 2, L.lang)}",
                                                        f"de overlap van de twee teksten: {C.number(_j, 2, L.lang)}"],
                                              "alt": f"Een rij van {len(_agree)} vakjes, één per hashfunctie. {sum(_agree)} vakjes zijn gevuld, waar de signatures van de twee teksten gelijk zijn: een schatting van {C.number(sum(_agree) / len(_agree), 2, L.lang)}, voor een overlap van {C.number(_j, 2, L.lang)}."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om de near-duplicates uit een corpus te halen:

    1. Knip elke pagina in shingles van $k$ woorden, en geef elke shingle een getal met een hashfunctie.
    2. Maak voor elke pagina een signature: voor elk van veel hashfuncties het kleinste getal dat hij aan de shingles van de pagina geeft.
    3. Knip de signatures in banden; pagina's die in een hele band gelijk zijn, zijn kandidaten.
    4. Vergelijk elk kandidaatpaar op zijn overlap, en zet pagina's die minstens 0,8 delen in één groepje.
    5. Houd van elk groepje één pagina, en haal de rest weg.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 1 en de overlap voor twee korte teksten die in één woord verschillen, met reeksen van 2 woorden.
    """)
    return


@app.cell
def example_overlap():
    _a = "the cat sat on the mat and looked at the birds all day".split()
    _b = "the cat sat on the hat and looked at the birds all day".split()
    # 1. elke reeks van 2 woorden, als set
    _sa = set()
    for _i in range(len(_a) - 1):
        _sa.add(" ".join(_a[_i:_i + 2]))
    _sb = set()
    for _i in range(len(_b) - 1):
        _sb.add(" ".join(_b[_i:_i + 2]))
    # de overlap: de shingles die beide hebben, gedeeld door alle verschillende shingles van de twee
    print(len(_sa & _sb), "gedeeld van de", len(_sa | _sb), "samen:", round(len(_sa & _sb) / len(_sa | _sb), 3))
    print("alleen in tekst 1:", _sa - _sb)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Eén woord verschilt, en het zit in twee shingles: "the mat" en "mat and" zitten alleen in tekst 1, "the hat" en "hat and" alleen in tekst 2. De andere 10 zijn gedeeld, van de 14 samen: een overlap van 0,714. Met reeksen van 5 woorden, zoals in de demo, verandert één woord tot 5 shingles, dus korte teksten hebben een kleinere $k$ nodig. Met een kleine $k$ delen pagina's die niets met elkaar te maken hebben gewone reeksen zoals "one of the", dus pagina's hebben een grotere nodig.
    """)
    return


@app.cell(hide_code=True)
def text_numpy_rung():
    mo.md(r"""
    ### Een trede hoger: hetzelfde in NumPy

    Hetzelfde met arrays. Met NumPy werkt één hashfunctie op alle hashes van een pagina tegelijk, en vergelijk je twee signatures in één regel. De getallen hieronder zijn verzonnen, klein genoeg om te lezen.
    """)
    return


@app.cell
def example_numpy():
    import numpy as np

    _P = 2 ** 31 - 1
    _hashes = np.array([12, 7, 30])
    # één hashfunctie op elke hash tegelijk, en het kleinste getal ervan
    _values = (5 * _hashes + 3) % _P
    print(_values, _values.min())
    # twee signatures: het deel van de plekken dat gelijk is
    _sig1 = np.array([4, 18, 9, 27])
    _sig2 = np.array([4, 18, 11, 27])
    print((_sig1 == _sig2).mean())
    return


@app.cell(hide_code=True)
def text_after_numpy():
    mo.md(r"""
    | Python | NumPy |
    |---|---|
    | een loop over de hashes, met `(a * h + b) % P` | `(a * hashes + b) % P`, een array |
    | het kleinste bewaren in de loop | `.min()` |
    | de plekken tellen die gelijk zijn, en dan delen | `(sig1 == sig2).mean()` |

    `sig1 == sig2` geeft een array van `True` en `False`, en het gemiddelde daarvan telt `True` als 1: hier 3 van de 4, 0,75.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shingles():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Shingles

        Maak `shingles(text, k)` af: stap 1 van het recept. Hij geeft de set terug van alle reeksen van `k` woorden achter elkaar uit `text` in kleine letters, elk als één tekst met een spatie tussen de woorden. Een tekst met minder dan `k` woorden heeft er geen. De check roept `shingles("A dog ran after the Red ball", 3)` aan, en andere.
        """),
        L.task("Laat `shingles` elke reeks van `k` woorden geven, één keer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shingles():
    box_shingles = L.code_box("shingles", lines=8)
    box_shingles
    return (box_shingles,)


@app.cell(hide_code=True)
def check_shingles(box_shingles):
    result_shingles = L.check("shingles", box_shingles.value)
    result_shingles
    return (result_shingles,)


@app.cell(hide_code=True)
def hints_shingles():
    L.hints(
        "shingles",
        [
            "Het uitgewerkte voorbeeld doet het voor `k` = 2. Een reeks van `k` woorden begint op elke plek `i` waarna nog `k` woorden volgen.",
            "De laatste reeks begint bij woord `len(words) - k`. Tot waar moet de `range` lopen om die plek mee te nemen? De reeks vanaf plek `i` is een slice van `words`.",
        ],
        why="Een set bewaart elke shingle één keer, dus een pagina die een zin herhaalt, telt hem niet twee keer, en twee sets vergelijk je met `&` en `|`. Kleine letters maken \"The\" en \"the\" hetzelfde woord.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_minhash():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. MinHash in Python

        Maak `minhash(hashes, a, b)` af: stap 2 van het recept. `hashes` zijn de getallen van de shingles van een pagina; `a` en `b` bevatten de getallen van de hashfuncties, functie `i` met `a[i]` en `b[i]`. Hij geeft de signature terug, een lijst met voor elke functie de kleinste `(a[i] * h + b[i]) % P` over alle `h` in `hashes`, of `P` als die er niet zijn. De check roept `minhash([5, 9, 2], [3, 7], [1, 4])` aan, en andere.
        """),
        L.task("Laat `minhash` voor elke hashfunctie het kleinste getal bewaren dat hij geeft."),
    ])
    return


@app.cell(hide_code=True)
def exercise_minhash():
    box_minhash = L.code_box("minhash", lines=12)
    box_minhash
    return (box_minhash,)


@app.cell(hide_code=True)
def check_minhash(box_minhash):
    result_minhash = L.check("minhash", box_minhash.value)
    result_minhash
    return (result_minhash,)


@app.cell(hide_code=True)
def hints_minhash():
    L.hints(
        "minhash",
        [
            "Binnen de loop over de functies een tweede loop over `hashes`, zoals wanneer je zelf het kleinste getal van een lijst zoekt.",
            "Begin bij `smallest = P`: elke uitkomst van `% P` ligt eronder, dus de eerste vervangt hem.",
        ],
        why="Elke functie zet de shingles in zijn eigen volgorde; de kleinste is de eerste in die volgorde. Twee pagina's zijn op plek `i` gelijk als voor beide dezelfde shingle vooraan staat, en de kans daarop is ongeveer hun overlap.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_signature():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Signatures met NumPy

        Maak `signature(hashes, a, b)` af: dezelfde signature als je `minhash`, met alle hashfuncties op alle hashes tegelijk. `np.outer(h, a)` maakt een tabel met een rij per hash en een kolom per functie: `h[r] * a[c]` in rij `r` en kolom `c`. Daarna telt `+ b` bij elke kolom `b[c]` op. De check roept `signature([5, 9, 2], [3, 7], [1, 4])` aan, en andere; de demo roept hem 5.614 keer aan, met 128 functies.
        """),
        L.task("Laat `signature` de tabel van elke functie op elke hash uitrekenen, en het kleinste getal van elke kolom."),
    ])
    return


@app.cell(hide_code=True)
def exercise_signature():
    box_signature = L.code_box("signature", lines=12)
    box_signature
    return (box_signature,)


@app.cell(hide_code=True)
def check_signature(box_signature):
    result_signature = L.check("signature", box_signature.value)
    result_signature
    return (result_signature,)


@app.cell(hide_code=True)
def hints_signature():
    L.hints(
        "signature",
        [
            "Neem de regel van je `minhash` die `v` uitrekent, en zet `np.outer(h, a)` waar die `a[i] * h` heeft, en `b` waar die `b[i]` heeft: dat geeft elke `v` tegelijk.",
            "Het kleinste getal van elke kolom is `values.min(axis=...)`: welke as loopt langs de rijen omlaag?",
        ],
        why="As 0 loopt langs de rijen omlaag, over de hashes, dus `values.min(axis=0)` houdt één getal per kolom, per functie: de signature. NumPy doet de twee loops van je `minhash` in gecompileerde code, vele keren sneller.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### Wat je functies vinden in de opgeschoonde webpagina's

    De knop hieronder downloadt de webpagina's van [Tekst opschonen: quality filters](/course/open/llm-quality-filter?lang=nl), één keer, als je dat daar nog niet deed: de eerste {round(C.Q.hub.PINNED[C.C4_SAMPLE]['size'] / 1e6)} MB van een shard van C4 van vóór het opschonen. De course schoont ze op met zijn eigen versies van de filters van die les, en dan knipt je `shingles` elke pagina die overblijft in shingles van {C.K} woorden en geeft je `signature` hem {C.N_HASHES} getallen. Banding door de course, {C.BANDS} banden van {C.ROWS}, vindt de kandidaatparen, en elk kandidaatpaar wordt vergeleken op zijn overlap. Het duurt ongeveer tien seconden.
    """)
    return


@app.cell(hide_code=True)
def download_button_cell():
    download_button = mo.ui.run_button(label="Download de webpagina's")
    download_button
    return (download_button,)


@app.cell(hide_code=True)
def demo_dedup(download_button, box_shingles, box_signature, result_shingles, result_signature):
    _ready = result_shingles.state == "correct" and result_signature.state == "correct"
    mo.stop(not _ready, mo.md("*Maak eerst oefening 1 en 3 af: de demo gebruikt je `shingles` en je `signature`.*"))
    C.sample_or_stop(download_button, {
        "ask": "*Klik op de knop om de webpagina's te downloaden, 20 MB.*",
        "busy": "**Bezig met downloaden** · {done} van {total} MB",
        "no_net": "*Er is geen verbinding met Hugging Face. Probeer het opnieuw als je online bent.*",
        "not_pinned": "*Het bestand van Hugging Face is niet het bestand dat de course vastlegde, dus het is weggegooid. Meld het op GitHub, met de link onderaan de les.*",
        "no_room": "*Er is geen ruimte meer op de schijf voor de download. Maak ruimte, en klik opnieuw.*",
        "failed": "*De download is niet gelukt: {why}.*",
    })
    _sh = run_code(box_shingles.value).names["shingles"]
    _sg = run_code(box_signature.value).names["signature"]
    _r = C.find_duplicates(_sh, _sg)
    _named = [C.GROUPS[_p][L.lang] for _p in _r["places"][:4] if _p in C.GROUPS]
    _which = "" if not _named else " De grootste groepjes: " + "; ".join(_named) + "."
    mo.md(f"""Van de {C.number(_r['all_pairs'], 0, L.lang)} paren van {C.number(_r['pages'], 0, L.lang)} pagina's maakt banding {C.number(_r['candidates'], 0, L.lang)} kandidaten, en {C.number(_r['similar'], 0, L.lang)} daarvan delen minstens {C.number(C.SIMILAR, 1, L.lang)}. Bij die kandidaten zit het deel van de plekken waar twee signatures gelijk zijn gemiddeld {C.number(_r['error'], 3, L.lang)} naast de echte overlap. Samen vormen ze {_r['groups']} groepjes near-duplicates, het grootste van {_r['largest']} pagina's: {_r['removed']} pagina's kunnen weg, als er van elk groepje één blijft. Slechts {_r['exact_extra']} daarvan zijn exacte kopieën.{_which}""")
    return


@app.cell(hide_code=True)
def text_exercise_memorize():
    mo.md(r"""
    ### Oefening 4. Waarom kopieën weghalen?
    """)
    return


@app.cell(hide_code=True)
def choice_memorize_cell():
    choice_memorize = L.choice(
        "memorize",
        question="Eén zin van 61 woorden stond meer dan 60.000 keer in C4. Wat doet dat met een model dat erop getraind wordt?",
        options={
            "nothing": "Niets: een model leert patronen, geen zinnen",
            "right": "Het leert de zin uit het hoofd, en verspilt er stappen aan",
            "better": "Het leert die zin beter, en dat helpt het bij elke tekst die het leest",
        },
        correct="right",
        why={
            "nothing": "Modellen nemen wel over wat ze vaak zagen: modellen die getraind waren op corpora met veel kopieën, namen meer dan 1% van wat ze zonder prompt schreven woord voor woord over uit hun training data (Lee en anderen, 2022).",
            "better": "De 60.000e keer leert het niets nieuws; het kost stappen die naar tekst hadden kunnen gaan die het model nog niet zag. Na deduplicatie hadden modellen minder stappen nodig om net zo goed te leren.",
            "right": "Na deduplicatie namen modellen tien keer minder vaak tekst woord voor woord over, en haalden ze dezelfde resultaten in minder stappen (Lee en anderen, 2022). Kopieën van testvragen zijn nog erger: de volgende les.",
        },
    )
    choice_memorize
    return (choice_memorize,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De pagina's die door de filters van [Tekst opschonen: quality filters](/course/open/llm-quality-filter?lang=nl) komen, nu zonder hun near-duplicates; de sets van [Tuples, sets en sorteren](/course/open/py-tuples-sets?lang=nl), nu als shingles. De volgende les zoekt een bijzonder soort kopie: testvragen die in de training data zijn beland.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_shingles, result_minhash, result_signature, choice_memorize):
    L.wrap_up(result_shingles, result_minhash, result_signature, choice_memorize)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
