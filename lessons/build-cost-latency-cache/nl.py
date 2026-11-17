# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-cost-latency-cache. Route: course/route/lessen/bouwen.yaml. Checks, the calls of version 2, Granite's embeddings of the questions, the course's cache and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    from everything_ai.lesson import lesson

    L = lesson("build-cost-latency-cache", "nl")
    C = L.checks   # versie 2 van de assistent van de bank op de 120 testvragen, de embeddings die Granite ervan maakte, de cache en de plaatjes
    QUESTIONS = C.QUESTIONS           # de 120 testvragen van banking77-small, in de volgorde waarin ze binnenkwamen
    CALLS = C.CALLS                   # wat versie 2 met elke vraag deed: "question", "seconds", "prompt_tokens", "output_tokens", "reply"
    SECONDS = C.SECONDS               # de seconden van elke call naar het model
    EMBED_SECONDS = C.EMBED_SECONDS   # de seconden van elke vraag als embedding met Granite, in een eigen request
    VECTORS = C.VECTORS               # VECTORS[i] is de embedding van QUESTIONS[i], met lengte 1
    PRICE_IN, PRICE_OUT = C.PRICE_IN, C.PRICE_OUT   # verzonnen prijzen, in dollar per miljoen tokens


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Eén keer betalen voor hetzelfde begin](/course/open/build-prompt-caching?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_reuse():
    quick_reuse = L.choice(
        "quick-reuse",
        question="Wat gebruikt de server bij prompt caching opnieuw van een eerder request?",
        options={
            "answer": "Het hele antwoord op een eerdere vraag",
            "start": "Zijn werk aan een begin dat terugkomt",
            "written": "De tokens die het model de vorige keer schreef",
        },
        correct="start",
        why={
            "start": "De server bewaart zijn werk aan het begin van de prompt, de KV cache, en gaat vanaf daar verder. Het model leest de rest nog steeds en schrijft een nieuw antwoord.",
            "answer": "Dat bouwt deze les: een cache met antwoorden. Prompt caching bewaart het werk van de server aan een begin dat terugkomt; het model schrijft nog steeds een nieuw antwoord. Zie [Eén keer betalen voor hetzelfde begin](/course/open/build-prompt-caching?lang=nl).",
            "written": "Het model schrijft zijn antwoord elke keer opnieuw. Prompt caching bewaart het werk van de server aan een begin van de prompt dat terugkomt. Zie [Eén keer betalen voor hetzelfde begin](/course/open/build-prompt-caching?lang=nl).",
        },
    )
    quick_reuse
    return (quick_reuse,)


@app.cell(hide_code=True)
def text_why():
    _f = C.findings()
    mo.md(r"""
    ## Waarom

    In [Een eval set en regression tests](/course/open/build-eval-set?lang=nl) beantwoordde versie 2 van de assistent van de bank de {n} testvragen. Op de computer van de course duurde een antwoord {median} seconden bij de mediaan, en het langzaamste {slowest}: zo lang zou een klant hier wachten. De {n} calls lazen {pin} tokens en schreven er {pout}; bij de verzonnen prijzen van [Batches, keys en de rekening](/course/open/build-batch-keys-costs?lang=nl), {pi} en {po} dollar voor een miljoen tokens, is dat {per} dollar voor 1.000 vragen. Elke vraag werd helemaal betaald, terwijl de {n} over maar {topics} onderwerpen gingen, {each} vragen elk.

    In [Eén keer betalen voor hetzelfde begin](/course/open/build-prompt-caching?lang=nl) bewaarde de server zijn werk aan het begin van een prompt; het model las nog steeds elke vraag en schreef elk antwoord. Een cache met antwoorden slaat het model over: hij geeft een antwoord dat hij eerder gaf.

    **Een voorbeeld uit het dagelijks leven.** Bij een informatiebalie komen elke dag dezelfde paar vragen. Wie erachter zit, zoekt de openingstijden niet elke keer op, maar zegt wat er een uur eerder ook gezegd werd. Stelt iemand een vraag die alleen hetzelfde klinkt, dan is een goed antwoord van eerder nu het verkeerde antwoord.
    """.format(n=_f["n"], median=L.number(_f["median"], 2), slowest=L.number(_f["slowest"], 2), pin=C.num(_f["prompt_tokens"], L.lang),
               pout=C.num(_f["output_tokens"], L.lang), pi=C.PRICE_IN, po=C.PRICE_OUT, per=L.number(_f["per_1000"], 2),
               topics=len(set(C.LABELS)), each=_f["n"] // len(set(C.LABELS))))
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Betekenis |
    |---|---|
    | **latency** | de tijd van het versturen van een request tot het antwoord er is; voor een klant het wachten |
    | **exact cache** | bewaart antwoorden bij de precieze tekst van het request, en geeft er alleen een als dezelfde tekst terugkomt |
    | **semantic cache** | bewaart antwoorden bij de embedding van de vraag, en geeft er een als een nieuwe vraag in betekenis dicht genoeg bij een eerdere ligt: de similarity is minstens een drempel |

    Meet bij elke call twee dingen: wat hij kost, zijn tokens erin en eruit maal hun prijzen, en zijn latency. Kijk naar de latency van veel calls, niet van één. De mediaan zegt hoe lang een klant meestal wacht; p95, ongeveer het 0,95-kwantiel, zegt wat de langzaamste één op de twintig krijgt. Een gemiddelde verbergt allebei.

    Een cache staat vóór het model. Een exact cache zoekt de tekst op: dezelfde tekst krijgt hetzelfde antwoord, en dat kost niets. Een semantic cache maakt van de vraag een embedding, zoals in [Zoeken op betekenis](/course/open/build-retrieval-embeddings?lang=nl), en zoekt de dichtstbijzijnde vraag die hij eerder beantwoordde. Vanaf de drempel geeft hij het antwoord op die vraag. Elke hit bespaart een hele call, en elke hit geeft een antwoord dat voor een andere vraag geschreven is.
    """)
    return


@app.cell(hide_code=True)
def picture_idea():
    mo.Html(C.flow_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: een cache met antwoorden

    1. **Meten:** schrijf de seconden en de tokens van elke call op, zoals een trace doet. Kijk naar de mediaan en p95, en naar wat 1.000 calls kosten.
    2. **Eerst exact:** zet de tekst in kleine letters zonder spaties aan de randen, en zoek hem op. Een hit kost niets en gaat altijd over dezelfde vraag.
    3. **Dan op betekenis:** maak van de vraag een embedding en zoek de dichtstbijzijnde vraag in de cache. Vanaf de drempel geef je zijn antwoord; anders vraag je het model en bewaar je zijn antwoord.
    4. **Bewaar met zorg:** alleen antwoorden die je eval set goedkeurde; de cache in deze les bewaart elk antwoord, zodat je ziet wat dat doet. Geen antwoorden over één klant, zoals de kaart of het saldo van die klant; antwoorden over wat verandert, zoals wisselkoersen, alleen met een tijd waarna ze verlopen.
    5. **Meet de cache als een versie:** zijn hits, wat ze bespaarden, en de eval set op de antwoorden die hij gaf.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _f = C.findings()
    _c = CALLS[0]
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `CALLS` bevat wat versie 2 deed met elk van de {n} vragen, in de volgorde waarin de course ze stuurde, alsof ze na elkaar binnenkwamen. `CALLS[0]` is *{q}*: die duurde {s} seconden, las {pt} tokens en schreef er {ot}. `SECONDS` bevat de seconden van alle {n}: samen {total}, {median} bij de mediaan, {p95} bij p95 en {slowest} op zijn langst. Dat is de latency van alleen het model, op de computer van de course met {threads} threads; een server ver weg doet er de weg heen en terug bij.

    De course maakte van elke vraag ook een embedding met Granite Embedding, het embedding model van de server van de course, in een eigen request: `EMBED_SECONDS`. Dat duurde {em} seconden bij de mediaan, ongeveer {times} keer zo snel als een antwoord. `VECTORS` bevat die embeddings, een rij per vraag, elk met lengte 1, dus het inproduct van twee rijen is hun cosine similarity.

    Wat één call kost, is het aantal prompt tokens maal `PRICE_IN` plus het aantal output tokens maal `PRICE_OUT`, gedeeld door een miljoen: voor `CALLS[0]` is dat {one} dollar. Alle {n} samen kostten {all} dollar.
    """.format(n=_f["n"], q=_c["question"], s=L.number(_c["seconds"], 2), pt=_c["prompt_tokens"], ot=_c["output_tokens"],
               total=L.number(_f["total"], 1), median=L.number(_f["median"], 2), p95=L.number(_f["p95"], 2), slowest=L.number(_f["slowest"], 2),
               threads=_f["threads"], em=L.number(_f["embed_median"], 3), times=C.num(round(_f["median"] / _f["embed_median"], -2), L.lang),
               one=L.number(C.dollars(_c["prompt_tokens"], _c["output_tokens"]), 6), all=L.number(_f["dollars"], 4)))
    return


@app.cell(hide_code=True)
def text_measured():
    _f = C.findings()
    _r = _f["runs"][C.THRESHOLD]
    _low = _f["runs"][0.75]
    _i, _j, _s = _r["other"][1]
    mo.md(r"""
    ### Wat de course mat

    **Exact:** onder de {n} testvragen vindt een exact cache geen enkele hit, want banking77-small bevat elke tekst één keer: de course liet herhalingen weg toen hij de set maakte. In een app komt dezelfde tekst wel terug: een knop met een vaste vraag, een klant die een vraag twee keer stuurt, de eval set nog eens op dezelfde versie. Daar antwoordt een exact cache meteen en is het antwoord nooit voor een andere vraag.

    **Op betekenis:** vanaf een similarity van {t} beantwoordde de semantic cache van de course {h} van de {n} vragen met een eerder antwoord. {saved} van de {total} seconden van het model en {tok} van de {alltok} tokens waren niet nodig. {o} van de {h} kregen het antwoord op een vraag over een ander onderwerp: *{qi}* kreeg het antwoord op *{qj}*, een vraag over een geblokkeerde pincode, bij een similarity van {s}.

    De eval set keurde {cp} van de {h} antwoorden uit de cache goed, en {fp} van de eigen antwoorden van het model op dezelfde vragen. De eval set kijkt naar de vorm van een antwoord, het onderwerp, de stappen en de lengte. Een antwoord uit de cache over hetzelfde onderwerp komt erdoor zoals het dat bij zijn eigen vraag deed: de eval set ziet niet of het bij deze vraag past. Een lagere drempel hergebruikt meer, en slechter: vanaf {lt} beantwoordde de cache {lh} vragen, en keurde de eval set {lcp} van zijn antwoorden goed tegen {lfp} van die van het model zelf.
    """.format(n=_f["n"], t=L.number(C.THRESHOLD, 2), h=len(_r["hits"]), saved=L.number(_r["seconds"], 1), total=L.number(_f["total"], 1),
               tok=C.num(_r["prompt_tokens"] + _r["output_tokens"], L.lang), alltok=C.num(_f["prompt_tokens"] + _f["output_tokens"], L.lang),
               o=len(_r["other"]), qi=QUESTIONS[_i], qj=QUESTIONS[_j], s=L.number(_s, 2), cp=_r["cached_pass"], fp=_r["fresh_pass"],
               lt=L.number(0.75, 2), lh=len(_low["hits"]), lcp=_low["cached_pass"], lfp=_low["fresh_pass"]))
    return


@app.cell(hide_code=True)
def picture_measured():
    mo.vstack([
        mo.Html(C.threshold_picture(C.THRESHOLD, L.lang)),
        L.watch_out("Een cache geeft het antwoord dat hij bewaarde, ook als de feiten intussen veranderd zijn. Geef antwoorden over iets wat verandert een vervaltijd mee, en houd antwoorden over één klant uit de cache."),
    ])
    return


@app.cell(hide_code=True)
def text_p95():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Het langzame eind

        `p95(seconds)` moet de tijd geven waar 95 van de 100 calls onder blijven. Sorteer de tijden van kort naar lang en neem die op 95 procent van de lijst: bij 120 tijden is dat plek 114, geteld vanaf 0, dus 5 tijden zijn langer. De startcode sorteert ze al in `in_order`, en geeft de kortste. Het plaatje onder de check gebruikt je functie op `SECONDS`, de calls naar het model, en op `EMBED_SECONDS`, de vragen als embedding voor de cache.
        """),
        L.task("Geef de tijd op 95 procent van `in_order`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_p95():
    box_p95 = L.code_box("p95", lines=5)
    box_p95
    return (box_p95,)


@app.cell(hide_code=True)
def check_p95(box_p95):
    result_p95 = L.check("p95", box_p95.value)
    result_p95
    return (result_p95,)


@app.cell(hide_code=True)
def hints_p95():
    L.hints(
        "p95",
        [
            "De mediaan is de tijd halverwege de gesorteerde lijst; p95 is de tijd op 95 procent ervan.",
            "De plek is een deel van `len(seconds)`, met `int` een heel getal gemaakt.",
            "Maak van de laatste regel `return in_order[int(len(seconds) * ___)]`.",
        ],
        why="""```python
def p95(seconds):
    in_order = sorted(seconds)
    return in_order[int(len(seconds) * 0.95)]
```

De p95 van het model ligt zo'n 2 seconden boven zijn mediaan; een vraag als embedding voor de cache duurt bij p95 zo'n twee honderdste seconde. Een hit is niet een beetje sneller: hij slaat het langzame deel over.""",
    )
    return


@app.cell(hide_code=True)
def text_exact():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Dezelfde vraag nog eens

        `exact_hits(questions)` moet de vragen tellen waarvan de tekst al eerder kwam, zoals een exact cache de tekst ziet: in kleine letters, zonder spaties aan de randen. De startcode maakt die `key` voor elke vraag en bewaart hem in `seen`, een set, maar zoekt er nooit een op. `x in a_set` is `True` als `x` in de set zit. Het plaatje laat je functie los op drie reeksen vragen.
        """),
        L.task("Tel een hit als de key al in `seen` zit, voordat je hem bewaart."),
    ])
    return


@app.cell(hide_code=True)
def exercise_exact():
    box_exact = L.code_box("exact_hits", lines=9)
    box_exact
    return (box_exact,)


@app.cell(hide_code=True)
def check_exact(box_exact):
    result_exact = L.check("exact_hits", box_exact.value)
    result_exact
    return (result_exact,)


@app.cell(hide_code=True)
def hints_exact():
    L.hints(
        "exact_hits",
        [
            "Eerst opzoeken, dan bewaren: andersom vindt elke vraag zichzelf.",
            "Zet een `if` vóór `seen.add(key)`.",
            "Schrijf `if key in seen:`, met `hits = ___` een stap verder ingesprongen.",
        ],
        why="""```python
def exact_hits(questions):
    seen = set()
    hits = 0
    for question in questions:
        key = question.lower().strip()
        if key in seen:
            hits = hits + 1
        seen.add(key)
    return hits
```

De tweede keer dat dezelfde 120 binnenkomen, is elke vraag een hit. De eerste twee balken blijven op 0 omdat de course herhalingen uit banking77-small liet; waar dezelfde tekst wel terugkomt, zoals bij een knop met een vaste vraag, is elke herhaling een hit.""",
    )
    return


@app.cell(hide_code=True)
def text_closest():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De dichtstbijzijnde eerdere vraag

        `closest(vector, vectors)` moet de plek geven van de rij van `vectors` die het meest op `vector` lijkt, met de similarity erbij. De rijen hebben lengte 1, dus hun inproducten met `vector` zijn de similarities; de startcode heeft ze al in `scores`, en geeft plek 0. Het plaatje laat de semantic cache van de course met jouw functie over de 120 vragen lopen, vanaf een similarity van {t}.
        """.format(t=L.number(C.THRESHOLD, 2))),
        L.task("Neem de plek van de hoogste score."),
    ])
    return


@app.cell(hide_code=True)
def exercise_closest():
    box_closest = L.code_box("closest", lines=6)
    box_closest
    return (box_closest,)


@app.cell(hide_code=True)
def check_closest(box_closest):
    result_closest = L.check("closest", box_closest.value)
    result_closest
    return (result_closest,)


@app.cell(hide_code=True)
def hints_closest():
    L.hints(
        "closest",
        [
            "Elke score is de similarity van één vraag in de cache; de dichtstbijzijnde heeft de hoogste.",
            "NumPy heeft een functie die de plek van het grootste getal in een array geeft; maak van de uitkomst een Python-`int`.",
            "Maak van de regel `best = int(___)`.",
        ],
        why="""```python
def closest(vector, vectors):
    scores = vectors @ vector
    best = int(np.argmax(scores))
    return best, float(scores[best])
```

De dichtstbijzijnde vraag is niet altijd dezelfde vraag: een klant die de nieuwe kaart niet kan vinden en een klant die de kaart kwijt is, schrijven bijna dezelfde woorden.""",
    )
    return


@app.cell(hide_code=True)
def text_stricter():
    mo.md(r"""
    ### Oefening 4. Een strengere cache
    """)
    return


@app.cell(hide_code=True)
def choice_stricter_cell():
    _f = C.findings()
    _r, _hi = _f["runs"][C.THRESHOLD], _f["runs"][0.9]
    _i, _j, _s = _hi["other"][0]
    choice_stricter = L.choice(
        "stricter",
        question="De cache van de course geeft een eerder antwoord vanaf een similarity van {t}. Wat verandert er vanaf 0,9?".format(t=L.number(C.THRESHOLD, 2)),
        options={
            "fewer": "Minder hits, en nog één voor een ander onderwerp",
            "none": "Minder hits, en geen ervan voor een ander onderwerp",
            "more": "Meer hits, omdat een strengere cache zekerder is",
        },
        correct="fewer",
        why={
            "fewer": "Vanaf 0,9 beantwoordde de cache {h} vragen in plaats van {h0}, dus het model beantwoordde er {d} meer. Eén kreeg nog het antwoord voor een ander onderwerp: *{qi}* kreeg het antwoord op *{qj}*, bij een similarity van {s}. Dicht bij elkaar in woorden is niet altijd hetzelfde onderwerp.".format(
                h=len(_hi["hits"]), h0=len(_r["hits"]), d=len(_r["hits"]) - len(_hi["hits"]), qi=QUESTIONS[_i], qj=QUESTIONS[_j], s=L.number(_s, 2)),
            "none": "Eén nog wel: *{qi}* kreeg het antwoord op *{qj}*, bij een similarity van {s}. Een hogere drempel laat minder verkeerde antwoorden door, niet geen.".format(
                qi=QUESTIONS[_i], qj=QUESTIONS[_j], s=L.number(_s, 2)),
            "more": "Een strengere cache hergebruikt minder: vanaf 0,9 beantwoordde hij {h} vragen in plaats van {h0}.".format(h=len(_hi["hits"]), h0=len(_r["hits"])),
        },
    )
    choice_stricter
    return (choice_stricter,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer andere drempels

    Verschuif de drempel en kijk wat de cache bespaart, hoeveel vragen het antwoord voor een ander onderwerp krijgen, en wat de eval set zegt van de antwoorden die hij gaf.
    """)
    return


@app.cell(hide_code=True)
def settings_threshold():
    threshold_slider = mo.ui.slider(0.7, 0.98, step=0.01, value=C.THRESHOLD, label="Drempel", show_value=True)
    threshold_slider
    return (threshold_slider,)


@app.cell(hide_code=True)
def demo_threshold(threshold_slider):
    mo.Html(C.threshold_picture(threshold_slider.value, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De seconden en tokens van elke call zijn wat een trace van [Zien wat er gebeurde](/course/open/build-tracing?lang=nl) bij elke span opschrijft. Een cache is een wijziging als elke andere: meet hem met de eval set van [Een eval set en regression tests](/course/open/build-eval-set?lang=nl) voordat klanten hem krijgen, en met gebruikers zoals in [Offline, online en wat het waard is](/course/open/build-eval-online?lang=nl). [Een LLM-app uitbrengen](/course/open/build-ship-app?lang=nl) zet de delen bij elkaar: instellingen in een bestand, logs, een versie van elke prompt, en een weg terug.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_stricter, result_closest, result_exact, result_p95):
    L.wrap_up(result_p95, result_exact, result_closest, choice_stricter)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
