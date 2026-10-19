# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les gpt-attention-maps. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("gpt-attention-maps", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Attention weights uit inproducten](/course/open/attn-scores?lang=nl) en [Multi-head attention](/course/open/attn-multi-head?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sum():
    quick_sum = L.choice(
        "quick-sum",
        question="Tot hoeveel tellen de attention weights in één rij van een head op?",
        options={
            "positions": "Het aantal posities in de rij",
            "right": "1",
            "free": "Niets vasts: het zijn scores",
        },
        correct="right",
        why={
            "positions": "Softmax maakt van de scores van een rij weights die samen 1 zijn. Zie [Attention weights uit inproducten](/course/open/attn-scores?lang=nl).",
            "free": "De scores kunnen alles zijn; softmax maakt er weights van die samen 1 zijn. Zie [Attention weights uit inproducten](/course/open/attn-scores?lang=nl).",
            "right": "Elke rij is een gewogen gemiddelde: hij verdeelt 1 over de posities die hij mag zien. Deze les leest die rijen.",
        },
    )
    quick_sum
    return (quick_sum,)


@app.cell(hide_code=True)
def quick_check_count():
    quick_count = L.choice(
        "quick-count",
        question="Een GPT met `n_layers` blocks van elk `h` heads leest één tekst. Hoeveel matrices met attention weights maakt hij?",
        options={
            "heads": "`h`: alle blocks delen dezelfde heads",
            "blocks": "`n_layers`: één per block",
            "right": "`n_layers` × `h`: één voor elke head",
        },
        correct="right",
        why={
            "heads": "Elk block heeft zijn eigen attention met zijn eigen weights. Zie [Multi-head attention](/course/open/attn-multi-head?lang=nl).",
            "blocks": "Elke head van een block rekent zijn eigen weights uit, naast elkaar. Zie [Multi-head attention](/course/open/attn-multi-head?lang=nl).",
            "right": "De GPT van de course heeft 5 blocks van 4 heads: 20 matrices voor elke tekst, en deze les bekijkt ze.",
        },
    )
    quick_count
    return (quick_count,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je GPT van [Project: je GPT op een boek trainen](/course/open/gpt-train-chars?lang=nl) schrijft tekst die op Shakespeare lijkt. Maar waar kijkt hij naar, als hij het volgende teken voorspelt? Zijn attention weights zijn de enige plek waar hij zegt hoeveel elke positie neemt van elke positie ervoor. Als figuur getekend laten ze zien dat heads elk een eigen taak krijgen. Zo in een model kijken is het begin van begrijpen waarom het doet wat het doet.
    """)
    return


@app.cell(hide_code=True)
def predict_heads():
    guess_heads = L.guess(
        "previous-heads",
        question="De GPT van de course heeft 5 blocks van 4 heads: 20 heads. Hoeveel daarvan leggen gemiddeld meer dan de helft van hun weight op het teken vlak ervoor?",
        options={
            "none": "Geen",
            "right": "Eén",
            "many": "Meer dan tien",
        },
        reveal=f"**Eén:** head 2 van block 0, die er ongeveer {L.number(C.previous_score(C.course_maps(C.DEMO_TEXT)[0][2].tolist()), 2)} van zijn weight legt, op elke tekst uit de toneelstukken. De andere 19 blijven onder de helft: de volgende leggen er ongeveer 0,4. Eén head is genoeg om het teken ervoor in elke positie te brengen; andere heads doen andere taken.",
    )
    guess_heads
    return (guess_heads,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Eén matrix per head.** Voor een tekst van $n$ tekens geeft een head een $n \times n$-matrix met attention weights $W$: rij $t$ bevat de weights die positie $t$ geeft aan de posities $0$ tot en met $t$, en telt op tot 1. Boven de diagonaal is alles 0: de causal mask.

        **Een attention-heatmap** tekent die matrix met een vakje per weight, donkerder naarmate hij groter is: de rijen zijn de posities die kijken, de kolommen de posities waar ze naar kijken. Patronen om naar te zoeken:

        - een donkere streep vlak onder de diagonaal: de head kijkt naar het teken vlak ervoor;
        - een donkere diagonaal: de head kijkt naar het teken zelf;
        - donkere kolommen: de head kijkt terug naar bepaalde tekens, zoals nieuwe regels;
        - een gelijkmatig grijs: de head neemt een gemiddelde van alles ervoor.

        **Meet het.** Met 20 heads en lange teksten wil je ook getallen. Hoeveel een head naar het teken ervoor kijkt, gemiddeld over de rijen die er een hebben, is zijn score $p$:
        """),
        L.formula(r"p = \frac{1}{n - 1} \sum_{t=1}^{n-1} [[hidden:W_{t, t-1}]]"),
        mo.md(r"""
        En hoeveel hij naar bepaalde tekens kijkt, zoals spaties en nieuwe regels: voor elke rij de som van zijn weights op posities met zo'n teken, gemiddeld over de rijen. Vergelijk dat met een head die zijn weight gelijk verdeelt: die komt ongeveer uit op het deel van zulke tekens onder de posities die hij ziet.

        De figuur: twee heads van de GPT van de course die een regel van Juliet lezen, met haar naam op de regel ervoor, zoals in de toneelstukken. Head 2 van block 0 kijkt naar het teken vlak ervoor; head 3 van block 3 kijkt terug naar de nieuwe regel, waar de regel begint. Blocks en heads tellen vanaf 0, zoals in Python.
        """),
        mo.Html(C.heatmap(C.course_maps()[0][2], C.TEXT, "Block 0, head 2", "Een heatmap van 25 bij 25 vakjes voor de tekst JULIET:, een nieuwe regel, en What's in a name? Bijna alle weight ligt op een streep vlak onder de diagonaal: elk teken kijkt naar het teken vlak ervoor.",
                          {"cols": "kijkt naar →", "rows": "↓ het teken dat kijkt"})),
        mo.Html(C.heatmap(C.course_maps()[3][3], C.TEXT, "Block 3, head 3", "Een heatmap van 25 bij 25 vakjes voor dezelfde tekst. Op de tweede regel liggen de donkerste vakjes in de kolom van de nieuwe regel: de meeste tekens van de regel kijken terug naar de nieuwe regel waar hij begint.",
                          {"cols": "kijkt naar →", "rows": "↓ het teken dat kijkt"})),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **attention-heatmap** | | een beeld van de attention weights van een head: een rij voor elke positie die kijkt, een kolom voor elke positie waar ze naar kijkt, donkerder voor een grotere weight | |
        | **previous-token head** | | een head die het grootste deel van zijn weight op het token vlak ervoor legt | |
        | $W_{t, t-1}$ | W t, t min 1 | de weight die positie $t$ geeft aan de positie ervoor | `W[t][t - 1]` |
        | $p$ | p | de score van een head op het teken ervoor: het gemiddelde van $W_{t, t-1}$ | `previous_score(W)` |
        """),
        L.watch_out("Een heatmap laat zien waar een head van neemt, niet waarom het model voorspelt wat het voorspelt. Wat een head neemt, is de value van een positie, en die verandert de output-projectie, het MLP en de heads van latere blocks nog. Een grote weight op een teken betekent niet dat de voorspelling ervan afhangt: om dat te weten, verander je de input of zet je de head uit, en kijk je wat de voorspelling doet."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om uit te zoeken waar de heads van een GPT naar kijken:

    1. Stuur een tekst door de GPT, en bewaar de matrix met attention weights van elke head van elk block.
    2. Teken elke matrix als heatmap, en zoek naar strepen, diagonalen en kolommen.
    3. Meet elke head: zijn score op het teken ervoor, en zijn deel op de tekens waar je benieuwd naar bent.
    4. Vergelijk met een head die zijn weight gelijk verdeelt, en kijk beter naar de heads die eruitspringen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 4: welk deel van zijn weight geeft een head die hem gelijk verdeelt aan spaties, in "to be or not"? Rij $t$ geeft $1 / (t + 1)$ aan elk van de $t + 1$ tekens die hij ziet, dus zijn deel op spaties is het aantal spaties daaronder gedeeld door $t + 1$. Het commentaar met `#` zegt wat elk deel is. De regel in de loop is nog niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_even():
    _text = "to be or not"
    _targets = " "
    _total = 0.0
    for _t in range(len(_text)):
        # de tekens die rij t ziet, en hoeveel daarvan een doel zijn
        _seen = _text[:_t + 1]
        _hits = sum([1 for _c in _seen if _c in _targets])
        _total = _total + 0.0  # maak af: _hits / len(_seen)
    # gemiddeld over de rijen
    print(round(_total / len(_text), 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Zoals hij is, drukt de cel `0.0` af. Afgemaakt drukt hij `0.234` af: een head die gelijk verdeelt, geeft hier ongeveer een kwart van zijn weight aan spaties, omdat 3 van de 12 tekens spaties zijn, en de eerste rijen er nog geen zien. Een head die spaties veel meer geeft dan zijn gelijke deel, heeft er een taak mee.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_previous():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. De score op het teken ervoor

        Maak `previous_score(W)` af: geef voor de attention weights `W` van één head, een lijst van rijen, de gemiddelde weight op het teken vlak ervoor terug, over de rijen die er een hebben, zoals in de formule. De check roept `previous_score([[1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [0.1, 0.6, 0.3]])` aan, wat 0,7 moet geven, en probeert 4 posities en 2.
        """),
        L.task("Laat `previous_score` het gemiddelde van `W[t][t - 1]` geven over de rijen vanaf 1."),
    ])
    return


@app.cell(hide_code=True)
def exercise_previous():
    box_previous = L.code_box("previous-score", lines=5)
    box_previous
    return (box_previous,)


@app.cell(hide_code=True)
def check_previous(box_previous):
    result_previous = L.check("previous-score", box_previous.value)
    result_previous
    return (result_previous,)


@app.cell(hide_code=True)
def hints_previous():
    L.hints(
        "previous-score",
        [
            "Rij 0 heeft geen teken ervoor, dus de rijen lopen van 1 tot en met `n - 1`: dat zijn `n - 1` rijen.",
            "In rij `t` staat het teken ervoor in kolom `t - 1`. Tel die op en deel door het aantal rijen.",
        ],
        why="`return sum([W[t][t - 1] for t in range(1, n)]) / (n - 1)`. Voor het voorbeeld: (0,8 + 0,6) / 2 = 0,7.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_share():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Het deel op bepaalde tekens

        Maak `share_on(W, text, targets)` af: tel voor elke rij `t` de weights op de posities `j` tot en met `t` op waarvan het teken `text[j]` in `targets` staat, en geef het gemiddelde over de rijen terug. De check roept `share_on([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.2, 0.7, 0.1, 0.0], [0.1, 0.1, 0.6, 0.2]], "a b,", " ,")` aan, wat 0,375 moet geven, en probeert andere teksten en doeltekens.
        """),
        L.task("Laat `share_on` de gemiddelde weight geven die een rij op de doeltekens legt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_share():
    box_share = L.code_box("share-on", lines=10)
    box_share
    return (box_share,)


@app.cell(hide_code=True)
def check_share(box_share):
    result_share = L.check("share-on", box_share.value)
    result_share
    return (result_share,)


@app.cell(hide_code=True)
def hints_share():
    L.hints(
        "share-on",
        [
            "Binnen de twee loops: staat het teken op positie `j` in de doelen? Tel dan de weight op die rij `t` aan positie `j` geeft.",
            "`if text[j] in targets:` en `W[t][j]`; deel aan het eind `total` door het aantal rijen.",
        ],
        why="`if text[j] in targets: total = total + W[t][j]`, en `return total / n`. Voor het voorbeeld staan de doelen op posities 1 en 3: de rijen geven 0, 0,5, 0,7 en 0,1 + 0,2 = 0,3, en 1,5 / 4 = 0,375.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### De heads van je GPT

    De cel hieronder stuurt de naam van Romeo en twee van zijn regels, 97 tekens, door je getrainde GPT van [Project: je GPT op een boek trainen](/course/open/gpt-train-chars?lang=nl), of door die van de course als je er geen getraind hebt. Hij meet alle heads met je twee functies, waar die goed zijn, en tekent de head met de hoogste score op het teken ervoor en die met het grootste deel op nieuwe regels, voor de eerste 24 tekens.
    """)
    return


@app.cell(hide_code=True)
def demo_heads(box_previous, box_share, result_previous, result_share):
    _r = C.demo(box_previous.value if result_previous.state == "correct" else None,
                box_share.value if result_share.state == "correct" else None)
    if _r["source"] is None:
        _out = mo.md("Er is geen getrainde GPT om naar te kijken: de checkpoint van de course ontbreekt in `data/`.")
    else:
        _gpt = "je GPT" if _r["source"] == "yours" else "de GPT van de course"
        _p_who = "je `previous_score`" if _r["yours"][0] else "de `previous_score` van de course"
        _s_who = "je `share_on`" if _r["yours"][1] else "de `share_on` van de course"
        _pl, _ph, _pv, _ = _r["prev"]
        _ml, _mh, _, _mv = _r["marks"]
        _out = mo.vstack([
            mo.md(f"In {_gpt}, gemeten met {_p_who} en {_s_who}: head {_ph} van block {_pl} heeft de hoogste score op het teken ervoor, {L.number(_pv, 2)}, en head {_mh} van block {_ml} het grootste deel op nieuwe regels, {L.number(_mv, 2)}. Een head die zijn weight gelijk verdeelt, zou ze {L.number(_r['uniform'], 2)} geven."),
            mo.Html(C.heatmap(_r["maps"][_pl][_ph][:24, :24], C.DEMO_TEXT[:24], f"Block {_pl}, head {_ph}", "Een heatmap van de head met de hoogste score op het teken ervoor.",
                              {"cols": "kijkt naar →", "rows": "↓ het teken dat kijkt"})),
            mo.Html(C.heatmap(_r["maps"][_ml][_mh][:24, :24], C.DEMO_TEXT[:24], f"Block {_ml}, head {_mh}", "Een heatmap van de head met het grootste deel op nieuwe regels.",
                              {"cols": "kijkt naar →", "rows": "↓ het teken dat kijkt"})),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_explain():
    mo.md(r"""
    ### Oefening 3. Wat een heatmap zegt
    """)
    return


@app.cell(hide_code=True)
def choice_explain_cell():
    choice_explain = L.choice(
        "not-explanation",
        question="In de GPT van de course legt head 2 van block 0 ongeveer 0,9 van zijn weight op het teken vlak ervoor. Wat zegt dat over de voorspelling van het volgende teken?",
        options={
            "right": "Alleen waar deze head van neemt, niet hoeveel de voorspelling ervan afhangt",
            "depends": "Dat de voorspelling van het volgende teken vooral afhangt van het teken vlak ervoor",
            "nothing": "Niets: attention weights zijn willekeurige getallen",
        },
        correct="right",
        why={
            "depends": "De head brengt de value van het teken ervoor binnen, maar 19 andere heads, de MLP's en de output-projecties vormen de voorspelling ook. De weight alleen zegt niet hoeveel het meetelt.",
            "nothing": "De weights zijn niet willekeurig: ze komen uit de queries en keys die de GPT geleerd heeft, en dezelfde head doet hetzelfde op elke tekst.",
            "right": "Een heatmap is een goede plek om te beginnen met kijken. Of het ertoe doet, ontdek je door de input te veranderen of de head uit te zetten, en te kijken wat de voorspelling doet.",
        },
    )
    choice_explain
    return (choice_explain,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De attention weights van [Attention weights uit inproducten](/course/open/attn-scores?lang=nl), met de causal mask van [De causal mask](/course/open/attn-causal-mask?lang=nl), nu gelezen uit de heads van [Multi-head attention](/course/open/attn-multi-head?lang=nl) in je getrainde GPT, en getekend als heatmaps, zoals de bigram-tabel van [De bigram-tabel als matrix van kansen](/course/open/lm-bigram-matrix?lang=nl). In grotere modellen hielpen zulke metingen, op tekst die zich herhaalt, om heads te vinden die kijken naar wat er volgde op de vorige keer dat het huidige token voorkwam, en dat in de voorspelling kopiëren.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_explain, result_previous, result_share):
    L.wrap_up(result_previous, result_share, choice_explain)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
