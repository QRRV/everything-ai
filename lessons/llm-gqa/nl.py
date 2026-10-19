# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les llm-gqa. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("llm-gqa", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Multi-head attention](/course/open/attn-multi-head?lang=nl) en [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_columns():
    quick_columns = L.choice(
        "quick-columns",
        question="Je attention heeft vectoren van $d = 12$ getallen in 3 heads. Welke kolommen van de keys gebruikt head 2, als je telt vanaf head 0?",
        options={
            "first": "Kolommen 0 tot en met 3",
            "right": "Kolommen 8 tot en met 11",
            "all": "Alle 12 kolommen, zoals elke head",
        },
        correct="right",
        why={
            "first": "Dat zijn de kolommen van head 0: `i * d_head` tot `(i + 1) * d_head`. Zie [Multi-head attention](/course/open/attn-multi-head?lang=nl).",
            "all": "Elke head werkt op zijn eigen kolommen, $d / h = 4$ ervan. Zie [Multi-head attention](/course/open/attn-multi-head?lang=nl).",
            "right": "Head $i$ heeft de kolommen $4i$ tot en met $4i + 3$: voor head 2 is dat 8 tot en met 11. Deze les laat meerdere query heads dezelfde key-kolommen gebruiken.",
        },
    )
    quick_columns
    return (quick_columns,)


@app.cell(hide_code=True)
def quick_check_tokens():
    quick_tokens = L.choice(
        "quick-tokens",
        question="Je GPT leest een venster van 64 tokens. Hoeveel keys maakt één head van één layer?",
        options={
            "one": "Eén, voor het laatste token",
            "square": "4.096, één voor elk paar tokens",
            "right": "64, één voor elk token",
        },
        correct="right",
        why={
            "one": "Elk token krijgt een query, een key en een value, zodat latere tokens ernaar kunnen kijken. Zie [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl).",
            "square": "De scores komen in paren, $64 \\times 64$; de keys komen één per token. Zie [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl).",
            "right": "En evenveel values. Een GPT die genereert, kan ze voor elk eerder token bewaren in plaats van ze opnieuw te berekenen, zoals de volgende les laat zien: deze les maakt dat goedkoper.",
        },
    )
    quick_tokens
    return (quick_tokens,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een GPT die tekst genereert, kan de keys en values van elk token dat hij gezien heeft bewaren, in elke head van elke layer, in plaats van ze opnieuw te berekenen, en leest ze dan allemaal voor elk nieuw token; de volgende les bouwt dat. Bij een lange tekst vertragen dat geheugen en het lezen ervan het genereren. **Grouped-query attention** laat meerdere query heads één key head en één value head delen. Llama 2 70B, Llama 3 en Mistral gebruiken het: de queries blijven verschillend, en de keys en values krimpen.
    """)
    return


@app.cell(hide_code=True)
def predict_cache():
    guess_cache = L.guess(
        "fewer-keys",
        question="Llama 2 70B heeft 64 query heads in elke layer, en maar 8 key heads en 8 value heads. Hoeveel minder geheugen kosten de keys en values die hij bewaart, vergeleken met 64 van elk?",
        options={
            "half": "De helft",
            "right": "8 keer minder",
            "much": "64 keer minder",
        },
        reveal="**8 keer minder.** Elke groep van 8 query heads deelt één key head en één value head: 8 van elk in plaats van 64. De queries worden niet bewaard, dus ze delen zou niets besparen.",
    )
    guess_cache
    return (guess_cache,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Stap 1: één key- en value-head per query head.** In multi-head attention heeft elke head zijn eigen queries, keys en values: $h$ van elk, elk $d_\text{head}$ kolommen breed.

        **Stap 2: deel ze in groepen.** **Multi-query attention** geeft alle $h$ query heads één key head en één value head. **Grouped-query attention** zit ertussen: $g$ key- en value-heads, elk gedeeld door een groep van $h / g$ query heads naast elkaar; $g$ is een deler van $h$, dus elke groep is even groot. Query head $i$ gebruikt key- en value-head
        """),
        L.formula(r"j = \left\lfloor \frac{i}{h / g} \right\rfloor"),
        mo.md(r"""
        De haken $\lfloor \, \rfloor$ ronden af naar beneden, op een heel getal. Met $h = 8$ en $g = 2$ gebruiken query heads 0 tot en met 3 key head 0, en 4 tot en met 7 key head 1. Met $g = h$ is dit weer multi-head attention, met $g = 1$ multi-query attention.

        **Stap 3: wat het bespaart.** De key- en value-projecties maken $g \cdot d_\text{head}$ getallen per token in plaats van $h \cdot d_\text{head}$: minder parameters, en vooral $h / g$ keer minder keys en values om te bewaren tijdens het genereren. De scores en de gewogen sommen van de values kosten evenveel FLOPs als eerst: elke query head vergelijkt zijn queries nog steeds met alle keys.

        **Stap 4: wat het kost.** Eén gedeelde head voor allemaal, zoals bij multi-query attention, kan wat kwaliteit kosten. De paper die grouped-query attention introduceerde, vond dat 8 groepen dicht bij de kwaliteit van multi-head attention komen, bij bijna de snelheid van multi-query attention.

        Het plaatje: in elke rij 8 query heads, en daaronder, met een sterkere rand, de key- en value-heads die ze gebruiken.
        """),
        mo.Html(C.heads_picture({"title": "Wie deelt keys en values", "mha": "multi-head", "gqa": "grouped, 2", "mqa": "multi-query",
                                 "lang": L.lang,
                                 "alt": "Drie rijen, elk met 8 query heads boven en hun key- en value-heads eronder. Multi-head: 8 key- en value-heads, één per query head. Grouped-query met 2 groepen: 2 key- en value-heads, elk gedeeld door 4 query heads naast elkaar. Multi-query: 1 key- en value-head voor alle 8."})),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **multi-query attention** | alle query heads delen één key head en één value head | |
        | **grouped-query attention** | groepen query heads naast elkaar delen één key head en één value head | |
        | $g$ | het aantal key- en value-heads | `n_kv_heads` |
        | $\lfloor x \rfloor$ | entier van $x$: $x$ naar beneden afgerond op een heel getal, zoals `int(x)` doet voor $x$ niet onder 0 | `int(x)` |
        """),
        L.watch_out("Een model wordt getraind met zijn eigen aantal key- en value-heads: de weights van een checkpoint leggen het vast. Om $g$ achteraf kleiner te maken, moet je heads samenvoegen en verder trainen: de paper nam het gemiddelde van de key- en value-heads van elke groep en trainde daarna nog 5% van de oorspronkelijke training."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo bereken je grouped-query attention met $h$ query heads en $g$ key- en value-heads:

    1. Maak de queries $h \cdot d_\text{head}$ breed, de keys en values $g \cdot d_\text{head}$.
    2. Zoek voor elke query head $i$ zijn key- en value-head: $j = \lfloor i / (h/g) \rfloor$, in Python `int(i / (n_heads / n_kv_heads))`.
    3. Draai causale attention van de kolommen van query head $i$ met de kolommen van key- en value-head $j$.
    4. Zet de outputs van de $h$ query heads naast elkaar.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Nieuw is welke head elke query head gebruikt: 8 query heads, 2 key- en value-heads.
    """)
    return


@app.cell
def example_groups():
    _n_heads, _n_kv_heads = 8, 2
    _group = int(_n_heads / _n_kv_heads)
    for _i in range(_n_heads):
        print("query head", _i, "gebruikt key- en value-head", int(_i / _group))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `_group` is 4: vier query heads per key- en value-head. `int` laat weg wat na de komma komt, dus `int(_i / _group)` stuurt query heads 0 tot en met 3 naar head 0 en 4 tot en met 7 naar head 1: groepen heads naast elkaar, zoals in de code van Llama.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_gqa():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. Grouped-query attention

        Maak `gqa(q, k, v, n_heads, n_kv_heads)` af: `q` heeft de queries van `n_heads` heads naast elkaar, shape (B, n, `n_heads * d_head`); `k` en `v` de keys en values van `n_kv_heads` heads, (B, n, `n_kv_heads * d_head`). Draai voor elke query head causale attention met de key- en value-head van zijn groep, zoals in [Het transformer block](/course/open/attn-transformer-block?lang=nl), en geef de outputs naast elkaar terug, (B, n, `n_heads * d_head`). De check roept `gqa(torch.randn(1, 3, 8), torch.randn(1, 3, 4), torch.randn(1, 3, 4), 4, 2)` aan, 4 query heads van 2 getallen die 2 key heads delen, en probeert 1 gedeelde head, 1 head per query head, en meer.
        """),
        L.task("Maak `gqa` af: query head i gebruikt de key- en value-head van zijn groep, met causale attention zoals eerder."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gqa():
    box_gqa = L.code_box("gqa", lines=17)
    box_gqa
    return (box_gqa,)


@app.cell(hide_code=True)
def check_gqa(box_gqa):
    result_gqa = L.check("gqa", box_gqa.value)
    result_gqa
    return (result_gqa,)


@app.cell(hide_code=True)
def hints_gqa():
    L.hints(
        "gqa",
        [
            "Stap 2 van het recept, zoals in het uitgewerkte voorbeeld, geeft de key- en value-head `j`; zijn kolommen lopen van `j * d_head` tot `(j + 1) * d_head`. De rest is de body van de loop van je causale attention, met die kolommen voor `k` en `v`.",
        ],
        why="Alleen de kolommen van de keys en values veranderen: `j = int(i / (n_heads / n_kv_heads))`, en zowel `k` als `v` gebruikt de kolommen van head `j`, terwijl `q` de kolommen van head `i` houdt.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_share():
    mo.md(r"""
    ### Oefening 2. Wie deelt
    """)
    return


@app.cell(hide_code=True)
def choice_share_cell():
    choice_share = L.choice(
        "share",
        question="Mistral 7B heeft 32 query heads en 8 key- en value-heads in elke layer. Welke query heads gebruiken key- en value-head 1?",
        options={
            "right": "Heads 4, 5, 6 en 7",
            "turns": "Heads 1, 9, 17 en 25",
            "eight": "Heads 8 tot en met 15",
        },
        correct="right",
        why={
            "right": "Groepen van $32 / 8 = 4$ heads naast elkaar: `int(i / 4)` is 1 voor heads 4 tot en met 7.",
            "turns": "Dat zouden heads zijn die elkaar afwisselen: 1, dan 9, dan 17. Een groep is heads naast elkaar: `int(i / 4)`.",
            "eight": "Dat zijn 8 heads: een groep heeft $32 / 8 = 4$ query heads.",
        },
    )
    choice_share
    return (choice_share,)


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ### Oefening 3. Waarom de keys en values
    """)
    return


@app.cell(hide_code=True)
def explain_why_keys():
    explain_keys_cell = L.explain(
        "why-keys",
        question="Grouped-query attention deelt de keys en values tussen query heads, maar geeft elke head zijn eigen queries. Waarom die twee delen en niet de queries, en waarom kost het weinig kwaliteit?",
        model_answer="**Waarom de keys en values:** terwijl een GPT genereert, bewaart hij de keys en values van elk eerder token, in elke head en layer, en leest hij ze allemaal voor elk nieuw token; de queries worden alleen voor het nieuwe token gemaakt en weggegooid. Keys en values delen maakt dus kleiner wat bewaard en gelezen wordt, met een factor $h / g$, en queries delen zou bijna niets besparen. **Waarom er weinig kwaliteit verloren gaat:** elke head stelt nog steeds zijn eigen vraag met zijn eigen queries, dus heads in een groep kunnen nog naar verschillende tokens kijken; alleen wat ze daar vinden, wordt gedeeld. Met een paar groepen, zoals in Llama 2 70B en Mistral, blijft de kwaliteit dicht bij multi-head attention.",
        points=[
            "je zegt dat keys en values van elk eerder token bewaard en gelezen worden tijdens het genereren, en queries niet",
            "je zegt dat ze delen dat geheugen met een factor h / g kleiner maakt",
            "je zegt dat elke head zijn eigen queries houdt, dus dat heads in een groep nog naar verschillende tokens kunnen kijken",
        ],
    )
    explain_keys_cell
    return (explain_keys_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De heads uit [Multi-head attention](/course/open/attn-multi-head?lang=nl), met groepen query heads die de kolommen van één key- en één value-head delen. De volgende les, *De KV-cache*, bewaart die keys en values tijdens het genereren, en telt wat dat geheugen kost.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_share, explain_keys_cell, result_gqa):
    L.wrap_up(result_gqa, choice_share, explain_keys_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
