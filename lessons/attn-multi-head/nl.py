# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les attn-multi-head. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("attn-multi-head", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [De causal mask](/course/open/attn-causal-mask?lang=nl) en [De scores schalen](/course/open/attn-scale?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_mask():
    quick_mask = L.choice(
        "quick-mask",
        question="Van welke posities kan het derde token van een tekst nemen, met de causal mask?",
        options={
            "right": "Van de eerste drie: zichzelf en de twee ervoor",
            "all": "Van alle posities: de mask verandert alleen de scores",
            "before": "Van de eerste twee: alleen die ervoor",
        },
        correct="right",
        why={
            "all": "De gemaskeerde scores worden min oneindig, dus de weights van latere posities zijn precies 0. Zie [De causal mask](/course/open/attn-causal-mask?lang=nl).",
            "before": "De diagonaal blijft: elke positie mag van zichzelf nemen. Zie [De causal mask](/course/open/attn-causal-mask?lang=nl).",
            "right": "Vandaag heeft elke head zijn eigen causale attention, met dezelfde mask.",
        },
    )
    quick_mask
    return (quick_mask,)


@app.cell(hide_code=True)
def quick_check_scale():
    quick_scale = L.choice(
        "quick-scale",
        question="Queries en keys van elk 16 getallen: door welk getal deelt geschaalde attention de scores?",
        options={
            "sixteen": "16: de lengte van een key",
            "right": "4: de wortel uit 16",
            "none": "Door niets: alleen lange keys hebben dat nodig",
        },
        correct="right",
        why={
            "sixteen": "Hij deelt door de wortel, $\\sqrt{d_k}$: delen door $d_k$ zou de scores te klein maken. Zie [De scores schalen](/course/open/attn-scale?lang=nl).",
            "none": "De scores spreiden ongeveer $\\sqrt{16} = 4$; delen houdt die spreiding op 1, bij elke lengte. Zie [De scores schalen](/course/open/attn-scale?lang=nl).",
            "right": "Vandaag wordt attention in heads gesplitst, smaller dan de hele vector, dus elke head deelt door de wortel uit zijn eigen lengte.",
        },
    )
    quick_scale
    return (quick_scale,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Eén attention geeft elke positie één rij weights: één manier om naar de tekst te kijken. Maar een woord heeft meerdere dingen tegelijk nodig. In "the black cat sat on the mat because it was tired" moet "it" vinden waar het naar verwijst, "sat" zijn onderwerp, en elk woord de woorden vlak ervoor. Eén softmax moet zijn weight over dat alles verdelen. Echte transformers draaien meerdere kleinere attentions naast elkaar: de kleinste GPT-2 heeft er 12 in elke layer.
    """)
    return


@app.cell(hide_code=True)
def predict_weights():
    guess_weights = L.guess(
        "weights-per-pair",
        question="Een zin gaat door attention met 4 heads naast elkaar. Hoeveel attention weights geeft één positie aan één andere positie?",
        options={
            "one": "1: de heads delen één softmax",
            "right": "4: één voor elke head",
            "sixteen": "16: één voor elk paar heads",
        },
        reveal="**4: één voor elke head.** Elke head rekent zijn eigen scores, zijn eigen softmax en zijn eigen gemiddelde uit, dus voor elk paar posities zijn er 4 weights, die heel verschillend kunnen zijn. Daar dienen de heads voor: vier manieren om tegelijk naar dezelfde tekst te kijken.",
    )
    guess_weights
    return (guess_weights,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Multi-head attention** draait $h$ **attention heads** naast elkaar. Elke head heeft zijn eigen queries, keys en values van $d_\text{head} = d / h$ getallen, en zijn eigen causale geschaalde attention. In de praktijk staan de matrices van alle heads naast elkaar in één matrix van $d$ bij $d$: $Q = \mathbf{X}\mathbf{W}_Q$ heeft $d$ kolommen, en elke head neemt zijn eigen blok van $d_\text{head}$ kolommen, van $Q$, $K$ en $V$ op dezelfde manier: head 1 het eerste blok, head 2 het volgende. De outputs van de heads komen weer naast elkaar, $n$ bij $d$, en een **output-projectie** $\mathbf{W}_O$, een lineaire layer van $d$ naar $d$, mengt ze:
        """),
        L.formula(r"\begin{aligned} [[hidden:\mathbf{H}_i]] &= \mathrm{softmax}\!\left(\frac{[[hidden:Q_i]] [[hidden:K_i]]^\top}{\sqrt{[[hyper:d_\text{head}]]}} + M\right) [[hidden:V_i]] \\ [[hidden:\mathbf{Y}]] &= [\,[[hidden:\mathbf{H}_1]] \; \cdots \; [[hidden:\mathbf{H}_h]]\,] \, [[param:\mathbf{W}_O]] + [[param:\mathbf{b}_O]] \end{aligned}"),
        mo.md(r"""
        Opsplitsen voegt geen parameters toe: de matrices zijn even groot als voor één head van $d$ getallen, en de vermenigvuldigingen kosten even veel werk. Wat verandert, is dat elke head eigen scores en een eigen softmax heeft, $h$ in plaats van één, dus zijn weight aan zijn eigen woorden kan geven. De kleinste GPT-2 heeft $d = 768$ in 12 heads van 64. In PyTorch zet `torch.cat(outputs, dim=1)` een lijst tensors naast elkaar, kolom na kolom.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **attention head** | | één causale attention op een eigen blok kolommen van $Q$, $K$ en $V$ | `causal_attention(q[:, a:b], k[:, a:b], v[:, a:b])` |
        | **multi-head attention** | | $h$ heads naast elkaar, met hun outputs naast elkaar gezet en gemengd door een output-projectie | `self.out(torch.cat(outputs, dim=1))` |
        | **output-projectie** | | een lineaire layer van $d$ naar $d$ na de heads, die hun outputs mengt | `nn.Linear(d, d)` |
        | $d_\text{head}$ | d head | het aantal kolommen van één head: $d / h$ | `d_head` |
        | `torch.cat` | torch cat | zet een lijst tensors aan elkaar, langs één axis | `torch.cat(outputs, dim=1)` |
        """),
        L.watch_out("Meer heads betekent niet meer parameters: dezelfde matrices van $d$ bij $d$ worden alleen in smallere heads geknipt. Maar meer heads maken elke head smaller: met te veel heads vergelijkt elke head maar een paar getallen. Niemand vertelt een head ook waar hij naar moet zoeken: wat elke head doet, komt uit het trainen."),
        mo.md(r"""
        De figuur: twee heads van "the cat sat down", met verzonnen queries en keys van elk één getal, en de causal mask. Head 1 zoekt een ding: vanaf "cat" neemt elk woord het meest van "cat". Head 2 zoekt een handeling: vanaf "sat" neemt elk woord het meest van "sat". In één attention zouden die twee één rij weights moeten delen.
        """),
        mo.vstack(C.heads_pictures({"one": "Head 1: zoekt een ding", "two": "Head 2: zoekt een handeling", "rows": "woord", "cols": "neemt van", "decimal": ",",
                                    "one_alt": "Causale attention weights van head 1 voor the cat sat down: vanaf cat neemt elk woord het meest van cat: 0,95, 0,91 en 0,87.",
                                    "two_alt": "Causale attention weights van head 2: cat neemt 0,5 van the en 0,5 van zichzelf; sat neemt 0,91 van zichzelf, en down 0,87 van sat."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Multi-head attention met $h$ heads, voor queries, keys en values van $d$ getallen per positie:

    1. De breedte van een head: $d_\text{head} = d / h$.
    2. Head `i`, met `i` uit `range(h)`: de kolommen van `i * d_head` tot `(i + 1) * d_head` van $Q$, $K$ en $V$, door causale attention.
    3. De outputs van alle heads naast elkaar: `torch.cat(outputs, dim=1)`.
    4. De output-projectie: een lineaire layer van $d$ naar $d$.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    3 posities met queries, keys en values van 4 getallen, in 2 heads van 2; de queries zijn hier ook de keys. `_causal_attention` is de causale attention van de vorige les. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 3 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_heads():
    import math
    import torch

    def _causal_attention(q, k, v):
        n, d_k = q.shape
        S = q @ k.T / math.sqrt(d_k)
        S = S.masked_fill(torch.tril(torch.ones(n, n)) == 0, float("-inf"))
        return torch.softmax(S, dim=1) @ v

    _Q = torch.tensor([[2.0, 0.0, 1.0, 0.0], [0.0, 2.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    _V = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.5, 0.5, 0.5, 0.5]])
    # 1. de breedte van een head
    _h = 2
    _d_head = int(_Q.shape[1] / _h)
    # 2. elke head op zijn eigen kolommen
    _outputs = []
    for _i in range(_h):
        _a, _b = _i * _d_head, (_i + 1) * _d_head
        _outputs.append(_causal_attention(_Q[:, _a:_b], _Q[:, _a:_b], _V[:, _a:_b]))
    # 3. de outputs naast elkaar
    _out = _outputs[0]  # maak af: torch.cat(_outputs, dim=1)
    print(_out)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Zoals de cel nu is, print hij alleen de output van head 1, met shape (3, 2): de tweede positie haalt 0,944 van zijn value uit zichzelf, net als in oefening 3 van [De causal mask](/course/open/attn-causal-mask?lang=nl). Afgemaakt print hij beide heads naast elkaar, met shape (3, 4). Head 2 ziet andere kolommen, `[1, 0]`, `[0, 1]` en `[1, 1]`, en geeft de tweede positie andere weights: 0,330 van de eerste positie en 0,670 van zichzelf. Eén attention over alle 4 kolommen zou de tweede positie één compromis geven, 0,076 en 0,924, voor beide helften.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_weights():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. De parameters van de heads
    """)
    return


@app.cell(hide_code=True)
def choice_weights_cell():
    choice_weights = L.choice(
        "head-weights",
        question="Vectoren van $d = 64$ getallen gaan in 4 heads van 16. Hoeveel parameters hebben de query-matrices van alle 4 heads samen?",
        options={
            "one": "1.024: 64 × 16, de matrix van één head",
            "four": "16.384: 4 × 64 × 64, een hele matrix per head",
            "right": "4.096: 64 × 64, net als voor één head van 64",
        },
        correct="right",
        why={
            "one": "Dat is één head. De 4 heads samen nemen alle 64 kolommen: 4 × 1.024.",
            "four": "Elke head neemt alleen zijn eigen 16 kolommen van één matrix van 64 bij 64, geen hele eigen matrix.",
            "right": "De 4 heads zijn 4 blokken van 16 kolommen van één matrix van 64 bij 64. Opsplitsen verandert hoe de scores berekend worden, niet hoeveel parameters er zijn.",
        },
    )
    choice_weights
    return (choice_weights,)


@app.cell(hide_code=True)
def text_exercise_multi_head():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Heads naast elkaar

        Maak `multi_head(Q, K, V, h)` af: één query, key en value per rij, met elk $d$ getallen; de functie geeft de outputs van $h$ causale heads naast elkaar, zonder de output-projectie. `causal_attention` uit de vorige les staat er al, net als stap 1 en de loop. Vervang de twee `...` volgens stap 2 en 3 van het recept. De check roept `multi_head` aan met 3 posities van 4 getallen in 2 heads, wat `[[1.0, 2.0, 3.0, 4.0], [3.679, 4.679, 6.7768, 7.7768], [6.0209, 7.0209, 7.0, 8.0]]` moet geven, en probeert 1, 3 en 4 heads.
        """),
        L.task("Laat `multi_head` de heads op hun eigen kolommen draaien en hun outputs naast elkaar zetten."),
    ])
    return


@app.cell(hide_code=True)
def exercise_multi_head():
    box_multi_head = L.code_box("multi-head", lines=19)
    box_multi_head
    return (box_multi_head,)


@app.cell(hide_code=True)
def check_multi_head(box_multi_head):
    result_multi_head = L.check("multi-head", box_multi_head.value)
    result_multi_head
    return (result_multi_head,)


@app.cell(hide_code=True)
def hints_multi_head():
    L.hints(
        "multi-head",
        [
            "De loop van het uitgewerkte voorbeeld, met `Q`, `K`, `V` en de namen van de functie, en stap 3 van het afgemaakte voorbeeld.",
            "`Q[:, a:b]` houdt elke rij, elke positie, en neemt de kolommen `a` tot `b`: het blok van één head.",
        ],
        why="Elke head ziet alle posities, maar alleen zijn eigen kolommen, dus hij vergelijkt andere getallen en geeft zijn eigen weights. `torch.cat(..., dim=1)` zet de blokken weer naast elkaar, in de volgorde van de heads.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_module():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Multi-head attention in PyTorch

        Maak de class `MultiHeadAttention(d, h)` af: in `__init__` de geleerde layers `self.query`, `self.key` en `self.value`, elk van `d` naar `d` getallen zonder bias terms, en `self.out`, de output-projectie van `d` naar `d`, in die volgorde gemaakt; in `forward` de heads op hun eigen kolommen, hun outputs naast elkaar, en daarvan `self.out`. `causal_attention` staat er weer, net als `self.h`, `self.d_head` en de eerste regel van `forward`. De check maakt de module na `torch.manual_seed(0)`, met `d = 6` en 2 heads, en vergelijkt zijn output voor 4 posities; daarna probeert hij 3 heads, en bekijkt hij de shapes van zijn parameters.
        """),
        L.task("Laat `MultiHeadAttention` de heads draaien en hun outputs mengen met een output-projectie."),
    ])
    return


@app.cell(hide_code=True)
def exercise_module():
    box_module = L.code_box("multi-head-module", lines=26)
    box_module
    return (box_module,)


@app.cell(hide_code=True)
def check_module(box_module):
    result_module = L.check("multi-head-module", box_module.value)
    result_module
    return (result_module,)


@app.cell(hide_code=True)
def hints_module():
    L.hints(
        "multi-head-module",
        [
            "In `__init__` vier `nn.Linear`-layers van `d` bij `d`; de eerste drie met `bias=False`, zoals in [Queries, keys en values](/course/open/attn-qkv?lang=nl).",
            "In `forward` de loop van oefening 2 met `q`, `k`, `v`, `self.h` en `self.d_head`, en `self.out` om `torch.cat` heen.",
        ],
        why="De drie layers maken $Q$, $K$ en $V$ voor alle heads tegelijk; de loop knipt ze in heads; `self.out` laat elk getal van de uitkomst de outputs van alle heads gebruiken. Bij $d = 6$ zijn dat $4 \\times 36 + 6 = 150$ parameters, met 2 heads of met 3.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De causale attention uit [De causal mask](/course/open/attn-causal-mask?lang=nl), met het schalen uit [De scores schalen](/course/open/attn-scale?lang=nl), nu in $h$ heads naast elkaar, op dezelfde matrices als één head. Er ontbreekt nog iets: attention vergelijkt inhoud, en behalve de mask zegt niets erin waar een woord staat. Het laatste woord van "dog bites man" en van "bites dog man" krijgt dezelfde nieuwe vector: het ziet dezelfde drie woorden, en attention kan niet zien in welke volgorde ze kwamen. In de volgende les, *Attention laten weten waar alles staat*, krijgt elke positie een vector van zijn eigen plek.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_weights, result_multi_head, result_module):
    L.wrap_up(choice_weights, result_multi_head, result_module)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
