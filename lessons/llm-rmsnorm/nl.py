# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les llm-rmsnorm. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("llm-rmsnorm", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Layer normalization](/course/open/nn-layernorm?lang=nl) en [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_ln():
    quick_ln = L.choice(
        "quick-ln",
        question="Layer normalization maakt van de rij $[1, 2, 3]$ een $\\hat{x}$, vóór $\\gamma$ en $\\beta$. Wat weet je van $\\hat{x}$?",
        options={
            "right": "Het gemiddelde is 0 en de variantie ongeveer 1",
            "sum": "De getallen tellen op tot 1, zoals na een softmax",
            "same": "Het is dezelfde rij: drie getallen hoef je niet te normaliseren",
        },
        correct="right",
        why={
            "right": "Layer normalization trekt het gemiddelde van de rij af en deelt door de standaardafwijking: ongeveer $[-1{,}22; 0; 1{,}22]$.",
            "sum": "Dat is softmax. Layer normalization centreert de rij en herschaalt hem. Zie [Layer normalization](/course/open/nn-layernorm?lang=nl).",
            "same": "Layer normalization werkt op elke rij van twee getallen of meer. Zie [Layer normalization](/course/open/nn-layernorm?lang=nl).",
        },
    )
    quick_ln
    return (quick_ln,)


@app.cell(hide_code=True)
def quick_check_where():
    quick_where = L.choice(
        "quick-where",
        question="Waar normaliseert je GPT uit [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl)?",
        options={
            "after": "Na elk block, op zijn output",
            "right": "Vóór attention en vóór het MLP in elk block, en nog één keer vóór de output head",
            "input": "Eén keer, op de embeddings, vóór het eerste block",
        },
        correct="right",
        why={
            "after": "Je blocks zijn pre-norm: elk onderdeel krijgt een genormaliseerde input, en telt zijn output op bij de residual stream. Zie [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl).",
            "right": "Twee normalisaties per block, en één aan het eind: 5 in een GPT van 2 blocks. RMSNorm komt precies op die plekken.",
            "input": "Elk block normaliseert de input van zijn attention en van zijn MLP. Zie [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl).",
        },
    )
    quick_where
    return (quick_where,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Llama, Mistral, Qwen en Gemma gebruiken geen layer normalization. Ze gebruiken **RMSNorm**: layer normalization zonder het gemiddelde. Het doet minder werk per token, heeft de helft van de parameters, en traint in grote language models ongeveer even goed. De modellen waarover je leest en de open recepten die je volgt, zijn ermee gebouwd, en de moderne GPT die je over een paar lessen bouwt ook.
    """)
    return


@app.cell(hide_code=True)
def predict_rms():
    guess_rms = L.guess(
        "rms-of-twos",
        question="Layer normalization maakt van de rij $[2, 2, 2, 2]$ nullen: elk getal is gelijk aan het gemiddelde. RMSNorm deelt de rij door de wortel uit het gemiddelde van zijn kwadraten. Wat maakt RMSNorm met $\\gamma = 1$ van $[2, 2, 2, 2]$?",
        options={
            "zeros": "$[0, 0, 0, 0]$",
            "right": "$[1, 1, 1, 1]$",
            "half": "$[0{,}5; 0{,}5; 0{,}5; 0{,}5]$",
            "same": "$[2, 2, 2, 2]$",
        },
        reveal="**$[1, 1, 1, 1]$.** Het gemiddelde van de kwadraten is 4, de wortel daaruit 2, en $2 / 2 = 1$. RMSNorm houdt de richting van de rij en stelt alleen zijn grootte bij: een rij met allemaal gelijke getallen blijft een rij met allemaal gelijke getallen.",
    )
    guess_rms
    return (guess_rms,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Stap 1: alleen herschalen.** De root mean square van een rij, zijn RMS, is de wortel uit het gemiddelde van zijn kwadraten. RMSNorm deelt elk getal van de rij erdoor, en vermenigvuldigt met een geleerde schaal $\gamma_j$ per feature: <!-- termen:ok -->
        """),
        L.formula(r"\begin{aligned} \text{RMS}(x) &= \sqrt{\tfrac{1}{d} \textstyle\sum_k x_k^2 + [[hyper:\varepsilon]]} \\ y_j &= [[param:\gamma_j]] \, \frac{x_j}{\text{RMS}(x)} \end{aligned}"),
        mo.md(r"""
        Zoals bij layer normalization voorkomt $\varepsilon$ dat een rij nullen door 0 gedeeld wordt, en begint $\gamma$ bij 1. Er is geen gemiddelde om af te trekken, en geen $\beta$.

        **Stap 2: wat het houdt, en wat het laat vallen.** Net als layer normalization trekt RMSNorm zich niets aan van de grootte van zijn input: $3x$ geeft hetzelfde als $x$. Dat houdt de getallen in een diep netwerk op een vaste grootte. Anders dan layer normalization let het wel op een verschuiving: $x + 5$ geeft iets anders dan $x$. Het artikel dat RMSNorm introduceerde, stelde dat het herschalen telt en dat het centreren weg kan, en in zijn experimenten trainden modellen zonder centreren ongeveer even goed. <!-- termen:ok -->

        **Stap 3: wat het bespaart.** Per rij berekent layer normalization twee getallen, het gemiddelde en de variantie, en trekt het gemiddelde van elke feature af; RMSNorm berekent er één, het gemiddelde van de kwadraten, en trekt niets af. Per normalisatie heeft layer normalization $2d$ parameters, RMSNorm $d$. Of het daardoor sneller is, hangt af van de code: `nn.LayerNorm` van PyTorch draait als één samengevoegde kernel, één programma dat de hele normalisatie in één keer doet, en op een CPU doet zijn `nn.RMSNorm` dat niet, dus daar kan layer normalization zelfs sneller zijn. Grote modellen gebruiken een samengevoegde RMSNorm.

        **Stap 4: waar het komt.** Precies op de plekken van layer normalization: vóór attention en vóór het MLP in elk block, en één keer vóór de output head. PyTorch heeft het als `nn.RMSNorm(d)`.

        Het plaatje: één rij, $x = [3, 5, 4, 6, 5, 7]$, en wat layer normalization en RMSNorm ervan maken.
        """),
        mo.Html(C.norms_picture({"title": "Eén rij, twee keer genormaliseerd", "x": "feature", "y": "waarde", "input": "x", "ln": "LayerNorm", "rms": "RMSNorm", "lang": L.lang,
                                 "alt": "Zes features. De input x gaat van 3 tot 7. Layer normalization, open stippen, centreert hem rond 0, van ongeveer −1,5 tot 1,5. RMSNorm, dichte stippen, houdt hem boven 0 en deelt hem door zijn RMS van ongeveer 5,2: van ongeveer 0,6 tot 1,4, dezelfde vorm als x."})),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **RMS** | de wortel uit het gemiddelde van de kwadraten | |
        | **RMSNorm** | elke rij delen door zijn RMS, en dan een geleerde schaal per feature | `nn.RMSNorm(d)` |
        """),
        L.watch_out("RMSNorm van een rij is niet hetzelfde als layer normalization ervan, ook niet met dezelfde $\\gamma$: alleen voor een rij met gemiddelde 0 komen ze overeen. De weights van een getraind model horen bij de normalisatie waarmee het getraind is."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo normaliseer je een rij met RMSNorm:

    1. Kwadrateer elk getal, en neem het gemiddelde langs de laatste axis, met behoud van die axis.
    2. Tel $\varepsilon$ op, en neem de wortel: de RMS van de rij.
    3. Deel de rij door zijn RMS, en vermenigvuldig elke feature met zijn $\gamma$.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `nn.RMSNorm` van PyTorch is nieuw: hier draait het naast `nn.LayerNorm` op twee rijen van 6 getallen, één keer herschaald en één keer verschoven.
    """)
    return


@app.cell
def example_invariance():
    import torch

    torch.manual_seed(0)
    _x = torch.randn(2, 6)
    _rms = torch.nn.RMSNorm(6, eps=1e-6)
    _ln = torch.nn.LayerNorm(6)
    print("RMSNorm, 3x hetzelfde als x:", torch.allclose(_rms(3 * _x), _rms(_x), atol=1e-5))
    print("RMSNorm, x + 5 hetzelfde als x:", torch.allclose(_rms(_x + 5), _rms(_x), atol=1e-5))
    print("layer normalization, x + 5 hetzelfde als x:", torch.allclose(_ln(_x + 5), _ln(_x), atol=1e-5))
    print("parameters:", sum(p.numel() for p in _rms.parameters()), "tegen", sum(p.numel() for p in _ln.parameters()))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Herschaald geeft RMSNorm hetzelfde; verschoven niet, terwijl layer normalization de verschuiving niet merkt. RMSNorm heeft 6 parameters, één $\gamma$ per feature; layer normalization heeft er 12, met ook een $\beta$ per feature. <!-- termen:ok -->
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rms():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. Je eigen RMSNorm

        Schrijf de class `RMSNorm(d, eps=1e-6)`: een weight `self.weight` van `d` getallen, $\gamma$, die bij 1 begint, en een `forward` die de laatste axis van zijn input normaliseert, van elke vorm. De check draait `RMSNorm(2)` op `torch.tensor([[3.0, 4.0], [1.0, 1.0]])`, wat ongeveer `[[0.8485, 1.1314], [1.0, 1.0]]` moet geven, en probeert een mini-batch tokens, een andere weight en een grote eps. De weight moet een `nn.Parameter` zijn, zoals de `.weight` van een `nn.Linear`: `nn.Parameter(t)` maakt van de tensor `t` een parameter van de module, een die `parameters()` vindt.
        """),
        L.task("Schrijf `RMSNorm`: een weight die bij 1 begint, en een `forward` die elke rij door zijn RMS deelt en hem schaalt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rms():
    box_rms = L.code_box("rmsnorm", lines=13)
    box_rms
    return (box_rms,)


@app.cell(hide_code=True)
def check_rms(box_rms):
    result_rms = L.check("rmsnorm", box_rms.value)
    result_rms
    return (result_rms,)


@app.cell(hide_code=True)
def hints_rms():
    L.hints(
        "rmsnorm",
        [
            "Het recept, met `x * x`, `.mean(dim=-1, keepdim=True)` en `torch.sqrt`; de weight als `nn.Parameter` om `torch.ones(d)`.",
        ],
        why="`keepdim=True` houdt één RMS per rij als kolom, zodat elke rij door zijn eigen RMS gedeeld wordt; eps hoort binnen de wortel; de weight, `torch.ones(d)`, maakt van een nieuwe normalisatie een zuivere herschaling.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### LayerNorm tegen je RMSNorm

    De knop traint je `GPT` uit [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl) twee keer op Tiny Shakespeare, met dezelfde start en dezelfde 400 mini-batches: één keer zoals hij is, en één keer met je `RMSNorm` op de plek van elke `nn.LayerNorm`. Een kleine GPT, 2 blocks met $d = 64$, op de CPU: het duurt ongeveer een minuut. Waar een module van jou ontbreekt, springt de versie van de course in. Hij werkt zodra oefening 1 goed is. Tijdens het trainen laat een grafiek de training loss van elke stap zien: eerst met LayerNorm, dan met je RMSNorm, met daarnaast gestippeld de run met LayerNorm. De laatste grafiek blijft boven de tabel staan.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train met LayerNorm en met RMSNorm")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_train(box_rms, result_rms, train_button):
    if result_rms.state != "correct":
        _out = mo.md("*Maak eerst oefening 1 af: deze cel gebruikt je `RMSNorm`.*")
    elif not train_button.value:
        _out = mo.md("*Druk op de knop om te trainen.*")
    else:
        _now = {}   # the live chart of each run, made at its first steps

        def _progress(_p):
            if _p["run"] not in _now:
                _rms = _p["run"] == "rms"
                _now[_p["run"]] = L.live({"title": "Je RMSNorm: de training loss van elke stap" if _rms else "LayerNorm: de training loss van elke stap", "x": "stap", "y": "loss",
                                          "alt": "De training loss van elke stap van de run met je RMSNorm tot nu toe, in grijs, en als dikkere lijn het gemiddelde over de laatste 10 stappen; gestippeld hetzelfde gemiddelde van de run met LayerNorm." if _rms else "De training loss van elke stap van de run met LayerNorm tot nu toe, in grijs, en als dikkere lijn het gemiddelde over de laatste 10 stappen."}, _p["steps"], window=10)
                if _rms and "layer" in _now:
                    _now["rms"].reference(list(zip(_now["layer"].xs, _now["layer"].means())), "LayerNorm", role="loss")
            _who = "LayerNorm" if _p["run"] == "layer" else "Je RMSNorm"
            _now[_p["run"]].extend(_p["fresh"], f"**{_who}** · stap {_p['step']} van {_p['steps']}")

        _m = C.demo(box_rms.value, _progress)
        _last = _now.get("rms") or _now.get("layer")   # a run that stops keeps its chart so far
        if _m == "error":
            _out = _last.stopped("*Het trainen stopte met een fout: werkt je GPT uit [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl) nog?*") if _last else mo.md("*Het trainen stopte met een fout: werkt je GPT uit [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl) nog?*")
        elif _m == "norms":
            _out = mo.md("*Je GPT heeft geen `nn.LayerNorm` om te vervangen: gebruikt hij `nn.LayerNorm`, zoals in [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl)?*")
        else:
            _n = lambda _x, _d=0: C.number(_x, _d, L.lang)
            _l, _r = _m["layer"], _m["rms"]
            _stand = (" De versie van de course sprong in voor: " + ", ".join(f"`{_s}`" for _s in _m["stand_ins"]) + ".") if _m["stand_ins"] else ""
            _pic = [mo.Html(_now["rms"].picture())] if "rms" in _now else []   # the RMSNorm run, LayerNorm dashed
            if _pic and "layer" in _now and _now["rms"].xs and _now["rms"].xs == _now["layer"].xs:   # both with every step a number
                _gap = max(abs(_a - _b) for _a, _b in zip(_now["rms"].means(), _now["layer"].means()))   # rounded up below
                _pic.append(mo.md(f"In de grafiek liggen de twee runs op elkaar: hun gemiddelden over 10 stappen lagen nooit meer dan {_n(_gap + 0.0005, 3)} uit elkaar, dus de dikke lijn van je RMSNorm bedekt de gestippelde van LayerNorm." if _gap < 0.01 else f"In de grafiek lagen de gemiddelden over 10 stappen van de twee runs hooguit {_n(_gap + 0.0005, 3)} uit elkaar."))
            _out = mo.vstack(_pic + [mo.md(f"""
    | | LayerNorm | je RMSNorm |
    |---|---|---|
    | validation loss | {_n(_l['val_loss'], 3)} | {_n(_r['val_loss'], 3)} |
    | parameters | {_n(_l['params'])} | {_n(_r['params'])} |
    | seconden | {_n(_l['seconds'], 1)} | {_n(_r['seconds'], 1)} |

    Je RMSNorm verving {_r['swapped']} normalization layers, en het model verloor daarmee {_n(_l['params'] - _r['params'])} parameters. In de eigen runs van de course met seeds 0, 1 en 2 eindigde RMSNorm hooguit 0,014 lager en hooguit 0,001 hoger, terwijl de loss van layer normalization door de seed alleen al meer dan 0,02 verschilde: een verschil van die grootte is ruis. De seconden verschillen weinig: de normalisaties zijn maar een klein deel van het werk, en een normalisatie die je zelf schreef, in een paar bewerkingen van PyTorch, is niet sneller dan de samengevoegde `nn.LayerNorm`.{_stand}
    """)])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_betas():
    mo.md(r"""
    ### Oefening 2. Wat een GPT bespaart
    """)
    return


@app.cell(hide_code=True)
def choice_betas_cell():
    choice_betas = L.choice(
        "betas",
        question="Een GPT met de maten van GPT-2 small, 12 blocks met $d = 768$, heeft 25 normalization layers: twee per block en één aan het eind. Je vervangt ze allemaal door RMSNorm. Hoeveel parameters verliest hij?",
        options={
            "both": "38.400: de $\\gamma$ en $\\beta$ van elke normalisatie",
            "none": "Geen: de normalisaties houden hun grootte",
            "right": "19.200: de $\\beta$ van elke normalisatie",
        },
        correct="right",
        why={
            "both": "RMSNorm houdt zijn $\\gamma$, één per feature: alleen $\\beta$ verdwijnt.",
            "none": "Elke layer normalization heeft een $\\beta$ van $d$ getallen die RMSNorm niet heeft.",
            "right": "$25 \\times 768 = 19.200$, van de ongeveer 124 miljoen: RMSNorm bespaart werk, geen parameters die ertoe doen.",
        },
    )
    choice_betas
    return (choice_betas,)


@app.cell(hide_code=True)
def text_exercise_mean():
    mo.md(r"""
    ### Oefening 3. Zonder het gemiddelde
    """)
    return


@app.cell(hide_code=True)
def explain_no_mean():
    explain_mean_cell = L.explain(
        "why-no-mean",
        question="RMSNorm laat weg wat layer normalization als eerste doet: het gemiddelde aftrekken. Waarom verliest een model daar weinig door, en wat wint het?",
        model_answer="**Wat blijft:** de deling door de RMS houdt elk token nog steeds op een vaste grootte, hoe groot de getallen die binnenkomen ook zijn; dat herschalen houdt een diep netwerk stabiel. **Waarom het gemiddelde weg kan:** de vaste grootte komt van het herschalen, niet van een gemiddelde van 0; het artikel dat RMSNorm introduceerde, vond dat modellen zonder het centreren ongeveer even goed trainen, en de grote language models van nu trainen goed met RMSNorm. **Wat het wint:** één getal per rij om te berekenen in plaats van twee, niets om af te trekken, en geen $\\beta$: minder werk per token, wat zich uitbetaalt in een samengevoegde kernel.",
        points=[
            "je zegt dat het herschalen, dat elk token op een vaste grootte houdt, blijft",
            "je geeft een reden waarom het centreren weinig toevoegt, zoals dat het herschalen alleen al de vaste grootte geeft, of dat modellen er zonder ongeveer even goed trainen",
            "je noemt de winst: één getal per rij in plaats van twee, en geen beta",
        ],
    )
    explain_mean_cell
    return (explain_mean_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De layer normalization van [Layer normalization](/course/open/nn-layernorm?lang=nl), zonder het gemiddelde en zonder $\beta$, op de plekken die ze heeft in [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl). De volgende les, *Moderne MLP's: GELU, SiLU en SwiGLU*, verandert een ander deel van elk block: zijn MLP.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_betas, explain_mean_cell, result_rms):
    L.wrap_up(result_rms, choice_betas, explain_mean_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
