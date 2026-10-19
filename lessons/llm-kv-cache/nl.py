# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les llm-kv-cache. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("llm-kv-cache", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Waar het geheugen heen gaat](/course/open/sys-memory-accounting?lang=nl) en [Grouped-query attention](/course/open/llm-gqa?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_weights():
    quick_weights = L.choice(
        "quick-weights",
        question="Je wilt een model van 13 miljard parameters in BF16 gebruiken, niet trainen. Hoeveel geheugen nemen zijn weights in?",
        options={
            "one": "13 GB",
            "right": "26 GB",
            "fp32": "52 GB",
        },
        correct="right",
        why={
            "one": "Dat is één byte per parameter. BF16 gebruikt 2 bytes per getal: zie [Waar het geheugen heen gaat](/course/open/sys-memory-accounting?lang=nl).",
            "fp32": "Dat is FP32, 4 bytes per parameter. BF16 gebruikt er 2: zie [Waar het geheugen heen gaat](/course/open/sys-memory-accounting?lang=nl).",
            "right": "2 bytes per parameter. Deze les telt erbij op wat een model naast zijn weights bewaart terwijl het schrijft.",
        },
    )
    quick_weights
    return (quick_weights,)


@app.cell(hide_code=True)
def quick_check_width():
    quick_width = L.choice(
        "quick-width",
        question="Een layer heeft 32 query heads en 8 key- en value-heads, elk 128 getallen breed. Hoeveel getallen maakt zijn key-projectie per token?",
        options={
            "right": "1.024",
            "query": "4.096",
            "one": "128",
        },
        correct="right",
        why={
            "query": "Dat is de breedte van de queries, 32 heads van 128. Er zijn maar 8 key heads: zie [Grouped-query attention](/course/open/llm-gqa?lang=nl).",
            "one": "Dat is één head. De key-projectie maakt alle 8 key heads naast elkaar: zie [Grouped-query attention](/course/open/llm-gqa?lang=nl).",
            "right": "8 key heads van 128 getallen, en evenveel values. Deze les bewaart ze, en telt wat dat kost.",
        },
    )
    quick_width
    return (quick_width,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Tekst genereren met je GPT](/course/open/gpt-generate?lang=nl) deed je GPT voor elk nieuw teken een hele forward pass over het venster. Het grootste deel van die pass doet opnieuw wat de stap ervoor al deed: door de causal mask kijkt een token nooit naar de tokens erna, dus in elke layer blijven zijn keys en values gelijk als de tekst groeit. De **KV cache** bewaart ze, zodat een stap alleen aan het nieuwe token werkt. Elk transformer-LLM gebruikt er een als het schrijft. Hij kost geheugen dat met elk token groeit, en dat geheugen bepaalt hoe lang de teksten kunnen zijn en hoeveel mensen één GPU tegelijk kan bedienen.
    """)
    return


@app.cell(hide_code=True)
def predict_cache():
    guess_cache = L.guess(
        "llama-cache",
        question="Llama 2 7B heeft 32 layers, elk met 32 heads van 128 getallen, en geen grouped-query attention. Het schrijft in BF16 en heeft tot nu toe 4.096 tokens tekst. Hoeveel geheugen neemt zijn KV cache in?",
        options={
            "small": "Ongeveer 20 MB",
            "right": "Ongeveer 2 GB",
            "weights": "Ongeveer 13 GB",
        },
        reveal="**Ongeveer 2 GB.** Elk token bewaart in elk van de 32 layers een key en een value van $32 \\times 128 = 4{.}096$ getallen van 2 bytes: $2 \\times 32 \\times 4{.}096 \\times 2$ bytes, 0,52 MB per token. Maal 4.096 tokens: 2,1 GB, voor één tekst. Een GPU die 8 van zulke teksten tegelijk schrijft, bewaart 17 GB cache, meer dan de 13,5 GB van de weights.",
    )
    guess_cache
    return (guess_cache,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Stap 1: wat gelijk blijft.** Met de causal mask kijkt het token op plek $t$ alleen naar de plekken $1$ tot en met $t$. Een token dat aan het eind bijkomt, verandert dus niets op de plekken ervoor: in elke layer zijn hun vectoren, en dus hun keys en values, wat ze waren. Zonder cache rekent elke stap ze allemaal opnieuw uit.

        **Stap 2: de cache.** Een KV cache bewaart in elke layer de keys en values van alle tokens tot nu toe: tensors met shape (B, t, d), één rij per token. De eerste pass, over het begin van de tekst, vult hem. Daarna stuurt een stap alleen het nieuwe token door het model. In elke layer maakt hij de query, key en value van het nieuwe token, zet de key en value aan het eind van de cache, en vergelijkt de query met elke key in de cache. Een mask is niet nodig: de cache bevat alleen eerdere tokens en het nieuwe.

        **Stap 3: wat het bespaart.** Zonder cache stuurt de stap met $t$ tokens ze alle $t$ door elke layer; met een cache één. De nieuwe query leest nog steeds alle $t$ keys en values, maar de projecties en de MLP's, het grootste deel van de FLOPs, werken op één token. De uitkomst is op afronding na dezelfde: de cache bespaart werk en verandert niets.

        Het plaatje: een begin van 3 tokens, en dan drie stappen die er elk één bijzetten. Opvallende vakjes gaan in die stap door het model; vage komen uit de cache.
        """),
        mo.Html(C.steps_picture({"title": "Vier stappen schrijven", "without": "zonder cache", "with": "met een KV cache", "step": "stap {n}",
                                 "computed": "door het model", "cached": "uit de cache",
                                 "alt": "Twee roosters van vier rijen, één rij per stap, met 3, 4, 5 en 6 vakjes, één per token. Zonder cache valt elk vakje van elke rij op: elke stap stuurt elk token door het model, 18 in totaal. Met een KV cache heeft de eerste rij 3 opvallende vakjes, het begin; in de andere rijen valt alleen het laatste vakje op en zijn de andere vaag, gelezen uit de cache: 6 in totaal."})),
        mo.md(r"""
        **Stap 4: wat het kost.** Elk token bewaart in elke layer een key en een value voor elk van de $g$ key- en value-heads. In bytes:
        """),
        L.formula(r"M = 2 \cdot L \cdot g \cdot d_\text{head} \cdot n \cdot B \cdot b"),
        mo.md(r"""
        met $L$ layers, $n$ tokens, $B$ teksten tegelijk en $b$ bytes per getal. Het groeit met elk token en elke tekst. En een stap leest het allemaal, samen met alle weights, voor het werk van één token per tekst: met weinig teksten tegelijk is schrijven memory-bound. Hier loont grouped-query attention: $h / g$ keer minder cache.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **KV cache** | de keys en values van elk eerder token, bewaard in elke layer terwijl een model schrijft | `k_cache`, `v_cache` |
        | $L$ | het aantal layers | `n_layers` |
        | $n$, $B$ | de tokens die per tekst bewaard worden, en de teksten tegelijk | `n_tokens`, `batch` |
        | $b$ | de bytes per getal: 2 in BF16, 4 in FP32 | `bytes_per_number` |
        """),
        L.watch_out("Een cache werkt alleen zolang de tekst in het venster past. Je GPT heeft een tabel van `block` plekken: wordt de tekst langer, dan schuift het venster bij elk nieuw token op, krijgt elk token een andere plek en veranderen alle keys en values. Vanaf dan heeft elke stap weer een hele pass nodig, en bespaart de cache niets."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo schrijf je met een KV cache:

    1. Stuur het begin van de tekst één keer door het model, en bewaar in elke layer de keys en values van al zijn tokens.
    2. Voor elk nieuw token, in elke layer: maak zijn query, key en value, en zet de key en value aan het eind van de cache van die layer, `torch.cat([k_cache, k], dim=1)`.
    3. Vergelijk zijn query met elke key in de cache: geschaald, door softmax, maal de values van de cache, zonder mask.
    4. Om het geheugen te tellen: $2 \cdot L \cdot g \cdot d_\text{head} \cdot n \cdot B \cdot b$ bytes.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Nieuw is een tensor die groeit: een cache die zonder tokens begint, en er per stap één key bij krijgt.
    """)
    return


@app.cell
def example_grow():
    import torch

    _k_cache = torch.zeros(1, 0, 2)  # één tekst, nog geen tokens, keys van 2 getallen
    for _step in range(3):
        _k = torch.randn(1, 1, 2)  # de key van het nieuwe token
        _k_cache = torch.cat([_k_cache, _k], dim=1)
        print("na stap", _step, "heeft de cache shape", tuple(_k_cache.shape))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `torch.zeros(1, 0, 2)` is een tensor zonder ook maar één token: de cache vóór de eerste stap. `torch.cat` met `dim=1` zet de nieuwe key achter de andere, op de as van de tokens: de cache groeit per stap met één rij, en de as van de teksten en de as van de getallen blijven zoals ze zijn.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_step():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. Eén stap met een cache

        Maak `attend_step(q, k, v, k_cache, v_cache)` af, één stap van één attention head met een cache. `q`, `k` en `v` zijn de query, key en value van het nieuwe token, shape (B, 1, d); `k_cache` en `v_cache` bevatten de keys en values van de eerdere tokens, (B, t, d). Zet de nieuwe key en value aan het eind van de caches, bereken de attention van het nieuwe token, en geef `out`, (B, 1, d), en de twee nieuwe caches terug. De check roept `attend_step(torch.randn(1, 1, 4), torch.randn(1, 1, 4), torch.randn(1, 1, 4), torch.randn(1, 2, 4), torch.randn(1, 2, 4))` aan, een nieuw token en een cache van 2 tokens, en probeert een lege cache, meer teksten tegelijk, een langere cache en andere breedtes.
        """),
        L.task("Maak `attend_step` af: zet de nieuwe key en value in de caches, en bereken dan de attention van het nieuwe token over elke key in de cache."),
    ])
    return


@app.cell(hide_code=True)
def exercise_step():
    box_step = L.code_box("attend-step", lines=12)
    box_step
    return (box_step,)


@app.cell(hide_code=True)
def check_step(box_step):
    result_step = L.check("attend-step", box_step.value)
    result_step
    return (result_step,)


@app.cell(hide_code=True)
def hints_step():
    L.hints(
        "attend-step",
        [
            "Stap 2 en 3 van het recept: het uitgewerkte voorbeeld laat zien hoe een cache groeit, voor de keys en voor de values. Dan attention zoals in causale attention, met `q` en de hele caches, maar zonder mask.",
        ],
        why="`k_cache = torch.cat([k_cache, k], dim=1)` en hetzelfde voor `v_cache`; dan `S = q @ k_cache.transpose(1, 2) / math.sqrt(q.shape[2])` en `out = torch.softmax(S, dim=2) @ v_cache`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_gb():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Wat de cache kost

        Schrijf `kv_cache_gb(n_layers, n_kv_heads, d_head, n_tokens, batch, bytes_per_number)`: het geheugen van een KV cache in GB, $10^9$ bytes. De check roept `kv_cache_gb(32, 8, 128, 8192, 1, 2)` aan, Llama 3 8B in BF16 met zijn venster van 8.192 tokens vol, wat ongeveer 1,07 moet geven, en probeert andere modellen, meer teksten tegelijk en FP32.
        """),
        L.task("Schrijf `kv_cache_gb`: de bytes van elke key en value die de cache bewaart, in GB."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gb():
    box_gb = L.code_box("kv-gb", lines=6)
    box_gb
    return (box_gb,)


@app.cell(hide_code=True)
def check_gb(box_gb):
    result_gb = L.check("kv-gb", box_gb.value)
    result_gb
    return (result_gb,)


@app.cell(hide_code=True)
def hints_gb():
    L.hints(
        "kv-gb",
        [
            "Stap 4 van het idee: één token bewaart een key en een value van `d_head` getallen voor elke key- en value-head, in elke layer. De rest is vermenigvuldigen, en delen door `1e9`.",
        ],
        why="`return 2 * n_layers * n_kv_heads * d_head * n_tokens * batch * bytes_per_number / 1e9`: voor Llama 3 8B is dat $2 \\times 32 \\times 8 \\times 128 \\times 2 = 131{.}072$ bytes per token.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fits():
    mo.md(r"""
    ### Oefening 3. Hoeveel teksten passen er
    """)
    return


@app.cell(hide_code=True)
def choice_fits_cell():
    choice_fits = L.choice(
        "fits",
        question="Een GPU van 80 GB bedient Llama 2 13B in BF16: 26 GB weights, 40 layers met 40 heads van 128 getallen, geen grouped-query attention. Hoeveel teksten van 4.096 tokens kan hij tegelijk schrijven, als alleen de weights en de KV caches geheugen innemen? Je `kv_cache_gb` kan helpen.",
        options={
            "all": "23",
            "keys": "32",
            "right": "16",
        },
        correct="right",
        why={
            "all": "Dan krijgt de cache alle 80 GB. De 26 GB weights moeten er ook in: er blijft 54 GB over, en één tekst neemt 3,36 GB in.",
            "keys": "Dan tel je alleen de keys, 1,68 GB per tekst. Elke layer bewaart per token ook een value: 3,36 GB per tekst.",
            "right": "Na de weights blijft er 54 GB over, en één tekst neemt 3,36 GB in. Met 8 key- en value-heads, zoals Llama 2 70B heeft, zou één tekst 5 keer minder innemen, en zouden er 80 teksten passen.",
        },
    )
    choice_fits
    return (choice_fits,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De loop van [Tekst genereren met je GPT](/course/open/gpt-generate?lang=nl), waarin de keys en values van eerdere tokens bewaard worden in plaats van opnieuw berekend, en het geheugen van [Waar het geheugen heen gaat](/course/open/sys-memory-accounting?lang=nl), geteld voor die keys en values. De volgende les, *Project: maak van je GPT een modern LLM*, zet RMSNorm, SwiGLU, RoPE, grouped-query attention en een KV cache in je GPT.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fits, result_gb, result_step):
    L.wrap_up(result_step, result_gb, choice_fits)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
