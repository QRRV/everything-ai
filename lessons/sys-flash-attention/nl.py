# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-flash-attention. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-flash-attention", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Geheugen of compute: wat remt je GPU af?](/course/open/sys-memory-roofline?lang=nl) en [De causal mask](/course/open/attn-causal-mask?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_memory():
    quick_memory = L.choice(
        "quick-memory",
        question="Waar op een GPU bewaart een groep threads data die hij vaak hergebruikt, dicht bij zijn cores?",
        options={
            "gpu": "In het GPU-geheugen",
            "right": "In shared memory",
            "host": "In host memory",
        },
        correct="right",
        why={
            "gpu": "Het GPU-geheugen is groot, maar elke keer lezen kost bandbreedte. Zie [Geheugen of compute: wat remt je GPU af?](/course/open/sys-memory-roofline?lang=nl).",
            "host": "Host memory is het eigen RAM van de computer, achter de trage PCIe-verbinding. Zie [Geheugen of compute: wat remt je GPU af?](/course/open/sys-memory-roofline?lang=nl).",
            "right": "Shared memory: op een RTX 4090 ongeveer 128 KB per streaming multiprocessor, klein en snel. FlashAttention houdt zijn blocks daar.",
        },
    )
    quick_memory
    return (quick_memory,)


@app.cell(hide_code=True)
def quick_check_causal():
    quick_causal = L.choice(
        "quick-causal",
        question="Welke scores gebruikt de softmax van rij $i$ bij causal attention?",
        options={
            "right": "De scores van de posities 0 tot en met $i$",
            "all": "Alle $n$ scores van de rij",
            "self": "Alleen de score van positie $i$ zelf",
        },
        correct="right",
        why={
            "all": "De mask zet de scores van de posities na $i$ op $-\\infty$, dus die krijgen weight 0. Zie [De causal mask](/course/open/attn-causal-mask?lang=nl).",
            "self": "Een positie mag nemen van zichzelf en van elke positie ervoor. Zie [De causal mask](/course/open/attn-causal-mask?lang=nl).",
            "right": "De scores na $i$ zijn $-\\infty$ en krijgen weight 0. In blocks kunnen hele blocks ervan worden overgeslagen.",
        },
    )
    quick_causal
    return (quick_causal,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Attention vergelijkt elk token met elk token ervoor: voor een tekst van $n$ tokens zijn dat $n \times n$ scores per head. Uitgerekend en bewaard groeit die matrix met het kwadraat van de lengte, en hem lezen en schrijven kost meer tijd dan de vermenigvuldigingen. **FlashAttention** bewaart hem nooit: het werkt de scores door in blocks die in shared memory passen, met een softmax die block voor block wordt bijgewerkt. Daardoor kunnen language models teksten van tienduizenden tokens lezen, en het is wat PyTorch gebruikt als je zijn attention aanroept.
    """)
    return


@app.cell(hide_code=True)
def predict_memory():
    guess_memory = L.guess(
        "score-memory",
        question="Een model leest een tekst van 32.768 tokens met 32 heads. Hoeveel geheugen nemen de matrices met scores van één layer in, bewaard in float32?",
        options={
            "one": "Ongeveer 1 GB",
            "ten": "Ongeveer 10 GB",
            "right": "Meer dan 100 GB",
        },
        reveal=f"**Meer dan 100 GB:** $32.768^2$ scores van 4 bytes is {C.size(32768 ** 2 * 4, 'nl')} per head, en 32 heads maken {C.size(32 * 32768 ** 2 * 4, 'nl')}: meer dan vijf keer de 24 GB van een RTX 4090. De tekst zelf, met $d = 4.096$, is {C.size(32768 * 4096 * 4, 'nl')}.",
    )
    guess_memory
    return (guess_memory,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Online softmax.** Voor één query neemt attention de scores $s_j$ met elke key, hun softmax, en de gewogen som van de values $\mathbf{v}_j$. Neem de scores in plaats van allemaal tegelijk één block per keer, en houd drie dingen bij: de grootste score tot nu toe $m$, de som $\ell$ van $e^{s_j - m}$, en de som $\mathbf{o}$ van $e^{s_j - m} \mathbf{v}_j$. Als er een block binnenkomt, kan zijn maximum groter zijn: het nieuwe maximum is
        """),
        L.formula(r"m' = \max(m, \max_j s_j)"),
        mo.md(r"""
        en elke term die tot nu toe is opgeteld, moet nu ten opzichte van $m'$ genomen worden: omdat $e^{s - m'} = e^{s - m} \, e^{m - m'}$, is dat één factor voor alles tegelijk. Tel dan het block erbij op:
        """),
        L.formula(r"\ell' = \ell \, e^{m - m'} + \sum_j e^{s_j - m'}"),
        L.formula(r"[[hidden:\mathbf{o}']] = [[hidden:\mathbf{o}]] \, e^{m - m'} + \sum_j e^{s_j - m'} [[hidden:\mathbf{v}_j]]"),
        mo.md(r"""
        Na het laatste block is de output $\mathbf{o} / \ell$: precies de output van softmax-attention, omdat elke term aan het eind ten opzichte van hetzelfde maximum genomen is. De factor $e^{m - m'}$ is hooguit 1, dus er loopt niets over.

        **FlashAttention** (Dao en anderen, 2022) doet dit voor blocks van queries tegelijk, in één fused kernel: het laadt blocks met queries, keys en values in shared memory, werkt ze daar af, en schrijft alleen de output, met $m$ en $\ell$, terug naar het GPU-geheugen. De $n \times n$-matrix bestaat nooit in het GPU-geheugen: het geheugen groeit met $n$, niet met $n^2$, en het verkeer naar het GPU-geheugen, dat attention afremt, wordt veel kleiner. Met de causal mask worden de blocks boven de diagonaal helemaal overgeslagen. FlashAttention-2 geeft elk block queries zijn eigen programma, dat alle blocks met keys en values afloopt, en verdeelt het werk zo beter over de GPU; latere versies stemmen het af op nieuwere GPU's.

        **Twee helften.** Twee programma's kunnen elk de helft van de keys nemen: elk eindigt met zijn eigen $m$, $\ell$ en $\mathbf{o}$, en de twee states worden samengevoegd door ze allebei naar het grootste maximum te schalen. Zo worden lange teksten over meer programma's verdeeld.

        De figuur: de matrix met scores in blocks van keys en queries, met de causal mask. De blocks op en onder de diagonaal worden uitgerekend; die erboven worden overgeslagen. Query-block 5 loopt zijn rij van links naar rechts af, en werkt na elk block $m$, $\ell$ en $\mathbf{o}$ bij.
        """),
        mo.Html(C.blocks_picture({"title": "Causal attention in blocks", "keys": "blocks met keys en values →", "queries": "blocks met queries",
                                  "walk": "m, ℓ, o", "note": "gestreept: overgeslagen",
                                  "alt": "Een rooster van 8 bij 8 vakjes. De vakjes op en onder de diagonaal zijn gevuld; die erboven zijn gestreepte randen. Rij 5 is donkerder gevuld, met een pijl van links naar rechts door zijn 5 vakjes, met de woorden m, l, o."})),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **online softmax** | | een softmax die block voor block wordt uitgerekend, met een lopend maximum en een lopende som die worden herschaald als het maximum groeit | |
        | **FlashAttention** | | attention in één fused kernel die door blocks met keys loopt in shared memory en de matrix met scores nooit bewaart | |
        | $m$, $\ell$, $\mathbf{o}$ | m, el, o | het lopende maximum, de lopende som van $e^{s - m}$, en de lopende gewogen som van de values | `m`, `l`, `o` |
        """),
        L.watch_out("Zonder mask doet FlashAttention dezelfde vermenigvuldigingen als attention die de matrix bewaart, en bij het trainen een paar meer: de backward pass rekent de scores opnieuw uit in plaats van ze terug te lezen. Het is sneller omdat het veel minder data naar en van het GPU-geheugen verplaatst. Met de causal mask slaat het ook de blocks boven de diagonaal over, die bewaarde attention uitrekent en daarna maskeert."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Attention voor één query, één block keys per keer:

    1. Begin met $m = -\infty$, $\ell = 0$ en $\mathbf{o} = \mathbf{0}$.
    2. Voor elk block: zijn scores, en het nieuwe maximum $m'$.
    3. Vermenigvuldig $\ell$ en $\mathbf{o}$ met $e^{m - m'}$; tel de termen $e^{s_j - m'}$ op bij $\ell$, en $e^{s_j - m'} \mathbf{v}_j$ bij $\mathbf{o}$.
    4. Na het laatste block: de output is $\mathbf{o} / \ell$.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 1 tot en met 3 zonder de values: de som $\ell$ waardoor de softmax deelt, voor de scores 1, 3, 2 en 5, in blocks van 2, vergeleken met de som in één keer met de grootste score eraf.
    """)
    return


@app.cell
def example_online_sum():
    import math

    _scores = [1.0, 3.0, 2.0, 5.0]
    _m, _l = float("-inf"), 0.0
    for _start in range(0, len(_scores), 2):
        _block = _scores[_start:_start + 2]
        _m_new = max(_m, max(_block))
        _l = _l * math.exp(_m - _m_new) + sum(math.exp(_s - _m_new) for _s in _block)
        _m = _m_new
        print("na het block", _block, "m =", _m, "l =", round(_l, 4))
    print("in één keer:", round(sum(math.exp(_s - max(_scores)) for _s in _scores), 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Na het eerste block is $m = 3$ en $\ell = e^{-2} + e^{0} \approx 1{,}1353$. Het tweede block brengt 5 mee: $\ell$ wordt vermenigvuldigd met $e^{3 - 5} \approx 0{,}1353$, en $e^{-3} + e^{0}$ komt erbij, wat 1,2034 geeft, hetzelfde als de som in één keer ten opzichte van 5. De weight van elke score is $e^{s - m} / \ell$; met de values geven dezelfde stappen de output.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_online():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. Online attention voor één query

        Maak `online_attention(q, K, V, block)` af: attention voor de query `q` over de keys in de rijen van `K` en de values in de rijen van `V`, met `block` keys per keer. De loop, de scores van een block en het nieuwe maximum staan er al; vervang de `...` door stap 3 van het recept. De check roept hem aan met een query van 2 getallen en 5 keys in blocks van 2, wat ongeveer [1,6059; 1,3563] moet geven, en probeert blocks van 1, één block met alle keys, en andere maten.
        """),
        L.task("Laat `online_attention` `l` en `o` naar het nieuwe maximum schalen, en elk block erbij optellen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_online():
    box_online = L.code_box("online-attention", lines=16)
    box_online
    return (box_online,)


@app.cell(hide_code=True)
def check_online(box_online):
    result_online = L.check("online-attention", box_online.value)
    result_online
    return (result_online,)


@app.cell(hide_code=True)
def hints_online():
    L.hints(
        "online-attention",
        [
            "De factor `math.exp(m - m_new)` voor wat je al hebt, en `torch.exp(s - m_new)` voor de termen van het block; `p @ V[start:start + block]` telt de values op, gewogen met die termen.",
        ],
        why="`scale = math.exp(m - m_new)`, `p = torch.exp(s - m_new)`, `l = l * scale + p.sum().item()` en `o = o * scale + p @ V[start:start + block]`. In het eerste block is `m` min oneindig, en `math.exp` van min oneindig is 0: er is nog niets om te herschalen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_merge():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Twee helften samen

        Twee programma's namen elk de helft van de keys van één query, en eindigden met de states `(m1, l1, o1)` en `(m2, l2, o2)`, met `o1` en `o2` lijsten. Maak `merge` af: geef de state `(m, l, o)` van alle keys samen terug. De check roept `merge(2.0, 1.5, [3.0, 1.0], 3.0, 1.0, [2.0, 0.0])` aan, wat ongeveer (3; 1,5518; [3,1036; 0,3679]) moet geven, en probeert andere states.
        """),
        L.task("Laat `merge` beide states naar het grootste maximum schalen en ze optellen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_merge():
    box_merge = L.code_box("merge", lines=8)
    box_merge
    return (box_merge,)


@app.cell(hide_code=True)
def check_merge(box_merge):
    result_merge = L.check("merge", box_merge.value)
    result_merge
    return (result_merge,)


@app.cell(hide_code=True)
def hints_merge():
    L.hints(
        "merge",
        [
            "Elke state werd genomen ten opzichte van zijn eigen maximum: zijn factor naar het nieuwe maximum is $e$ tot de macht zijn eigen maximum min het nieuwe.",
        ],
        why="`m = max(m1, m2)`, `a, b = math.exp(m1 - m), math.exp(m2 - m)`, `l = l1 * a + l2 * b` en `o = [o1[i] * a + o2[i] * b for i in range(len(o1))]`. In het voorbeeld wordt deel 1 geschaald met $e^{2 - 3} \\approx 0{,}3679$, en blijft deel 2 zoals het is.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Bewaard of in blocks, getimed

    De cel hieronder doet causal attention op één tekst van 2.048 tokens met 8 heads van 64 getallen, op je GPU als je die hebt, anders op je CPU: één keer met de matrix met scores uitgerekend, gemaskeerd en bewaard, en één keer met `F.scaled_dot_product_attention` van PyTorch, die in blocks werkt, zoals FlashAttention, waar de hardware dat toelaat, en de blocks boven de diagonaal overslaat. Het duurt een paar seconden.
    """)
    return


@app.cell(hide_code=True)
def demo_timed():
    _d = C.demo()
    _where = "je GPU" if _d["device"] == "cuda" else "je CPU"
    mo.md(
        f"Op {_where}, met de matrix bewaard: {L.number(_d['stored'] * 1000, 0)} ms; in blocks: {L.number(_d['fused'] * 1000, 0)} ms, "
        f"{L.number(_d['stored'] / _d['fused'], 1)} keer zo snel. De bewaarde matrices met scores nemen {C.size(_d['matrix'], 'nl')} in; "
        f"de twee outputs verschillen hooguit {L.number(_d['diff'] * 1e6, 1)} miljoenste, de afronding van float32."
    )
    return


@app.cell(hide_code=True)
def text_exercise_faster():
    mo.md(r"""
    ### Oefening 3. Waarom het sneller is
    """)
    return


@app.cell(hide_code=True)
def choice_faster_cell():
    choice_faster = L.choice(
        "why-faster",
        question="Zonder mask doet FlashAttention dezelfde vermenigvuldigingen als attention die de matrix met scores bewaart. Waarom is het sneller op een GPU?",
        options={
            "skip": "Het laat vermenigvuldigingen weg waarvan de weights klein zouden zijn",
            "right": "Het schrijft de scores en weights niet naar het GPU-geheugen en leest ze niet terug",
            "precision": "Het rekent altijd in een lagere precisie",
        },
        correct="right",
        why={
            "skip": "Elke score wordt uitgerekend: de output is precies die van softmax-attention. Alleen met de causal mask slaat het blocks over, die boven de diagonaal, en die zijn toch al gemaskeerd.",
            "precision": "Het rekent in hetzelfde getalformaat als andere attention; de snelheid komt van het verkeer dat het vermijdt.",
            "right": "Attention wordt afgeremd door geheugenverkeer, niet door vermenigvuldigingen. Bewaard gaan de $n \\times n$ scores een paar keer naar het GPU-geheugen en terug; in blocks blijven ze in shared memory.",
        },
    )
    choice_faster
    return (choice_faster,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De fused kernel van [Je eigen GPU kernel schrijven](/course/open/sys-triton-kernel?lang=nl), met de stabiele softmax een stap verder: een maximum dat onderweg wordt bijgewerkt. Het verkeer dat het vermijdt, is dat van [Geheugen of compute: wat remt je GPU af?](/course/open/sys-memory-roofline?lang=nl), en de blocks die het overslaat, zijn die van [De causal mask](/course/open/attn-causal-mask?lang=nl). In PyTorch kiest `F.scaled_dot_product_attention` zo'n kernel voor je.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_faster, result_merge, result_online):
    L.wrap_up(result_online, result_merge, choice_faster)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
