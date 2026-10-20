# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les llm-quantize. Route: course/route/lessen/taal.yaml. Checks and the measured numbers: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import torch
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-quantize", "nl")
    C = L.checks
    M = C.MEASURED            # the course's model, measured: its loss on the stories at each number of bits
    SIZE = C.course_sizes()   # and its megabytes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Minder bits: FP32, FP16 en BF16](/course/open/sys-number-formats?lang=nl) en [Capstone: je eigen kleine LLM pretrainen](/course/open/llm-pretrain-capstone?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_bf16():
    quick_bf16 = L.choice(
        "quick-bf16",
        question="BF16 en FP16 slaan allebei een getal op in 16 bits. Wat levert BF16 in, vergeleken met FP16?",
        options={
            "range": "Bereik: zijn getallen kunnen niet zo groot worden",
            "right": "Mantissebits: het bewaart minder cijfers",
            "sign": "De tekenbit: het slaat alleen getallen van 0 en meer op",
        },
        correct="right",
        why={
            "right": "BF16 besteedt 8 bits aan zijn exponent, zoals FP32, en houdt 7 mantissebits waar FP16 er 10 houdt: hetzelfde bereik als FP32, met minder cijfers.",
            "range": "Dat is FP16: met 5 exponentbits loopt het boven 65.504 over. BF16 houdt de 8 exponentbits van FP32. Zie [Minder bits: FP32, FP16 en BF16](/course/open/sys-number-formats?lang=nl).",
            "sign": "Alle drie de formaten houden een tekenbit: weights en gradients kunnen negatief zijn. Zie [Minder bits: FP32, FP16 en BF16](/course/open/sys-number-formats?lang=nl).",
        },
    )
    quick_bf16
    return (quick_bf16,)


@app.cell(hide_code=True)
def quick_check_megabytes():
    quick_megabytes = L.choice(
        "quick-megabytes",
        question="Het model van de course uit de capstone heeft 3.141.801 parameters, elk opgeslagen in FP32. Hoeveel megabyte nemen ze ongeveer in?",
        options={
            "right": "12,6 MB: 4 per parameter",
            "bits": "100,5 MB: 32 per parameter",
            "one": "3,1 MB: 1 per parameter",
        },
        correct="right",
        why={
            "right": "FP32 is 32 bits, 4 bytes, per getal: 3.141.801 × 4 bytes is 12,6 miljoen bytes.",
            "bits": "32 is het aantal bits van een FP32-getal; een byte bevat 8 bits, dus FP32 is 4 bytes. Zie [Minder bits: FP32, FP16 en BF16](/course/open/sys-number-formats?lang=nl).",
            "one": "Eén byte is 8 bits, en FP32 slaat elk getal op in 32: 4 bytes per parameter. Zie [Minder bits: FP32, FP16 en BF16](/course/open/sys-number-formats?lang=nl).",
        },
    )
    quick_megabytes
    return (quick_megabytes,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Om een model te draaien, houdt een computer zijn weights in het geheugen, en voor elk token dat het schrijft, leest het ze allemaal één keer. Een open model van 7 miljard parameters neemt in BF16 14 GB in: meer dan de meeste laptops kunnen missen, en meer dan veel GPU's hebben. En 14 GB lezen voor elk token is traag: zoals [Geheugen of compute: wat remt je GPU af?](/course/open/sys-memory-roofline?lang=nl) liet zien, is een model dat één token tegelijk schrijft memory-bound, met ongeveer twee bewerkingen voor elke weight die het leest. Sla elke weight op in 4 bits in plaats van 16, en het model neemt ongeveer een kwart van het geheugen in, en er is voor elk token ongeveer een kwart zoveel te lezen. De prijs is precisie: wat dat kost, meet je.

    **Een voorbeeld uit het dagelijks leven.** Een foto van je telefoon, opgeslagen als JPEG voor een website: tien keer kleiner, en je moet goed kijken om te zien wat er verloren ging. Sla hem te klein op, en hij valt uiteen in blokjes.
    """)
    return


@app.cell(hide_code=True)
def predict_loss():
    guess_loss = L.guess(
        "loss",
        question=f"Het model van de course uit de capstone heeft een loss van {C.number(M[(32, None)], 2, 'nl')} op verhalen waarop het nooit trainde. Opgeslagen in 4 bits per weight in plaats van 32, zoals open modellen meestal gedeeld worden, is het ongeveer zeven keer zo klein. Wat gebeurt er met die loss?",
        options={
            "small": "Hij stijgt een beetje, een paar honderdsten",
            "half": "Hij stijgt met ongeveer 0,5, tot boven 3",
            "broken": "Hij gaat boven 6: het model schrijft onzin",
        },
        reveal=f"**Hij stijgt een beetje:** tot {C.number(M[(4, C.GROUP)], 2, 'nl')}, als elke groep van {C.GROUP} weights een eigen schaal krijgt. Het model krimpt van {C.number(SIZE[(32, None)], 1, 'nl')} MB tot {C.number(SIZE[(4, C.GROUP)], 1, 'nl')} MB. Maar de andere twee antwoorden gebeuren ook, als je anders opslaat: met één schaal voor een hele matrix gaat de loss naar {C.number(M[(4, 'matrix')], 2, 'nl')}, en in 2 bits naar {C.number(M[(2, C.GROUP)], 2, 'nl')}. Hoe je de bits opslaat, telt net zo zwaar als hoeveel het er zijn.",
    )
    guess_loss
    return (guess_loss,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een rooster van hele getallen.** Met $b$ bits kun je $2^b$ verschillende hele getallen opslaan. Om 0 in het midden te houden, gebruikt deze vorm van quantization de hele getallen van $-q_{max}$ tot $q_{max}$, met $q_{max} = 2^{b-1} - 1$: van −7 tot 7 in 4 bits, van −127 tot 127 in 8. Elke weight $w$ wordt het hele getal $q$ dat het dichtst bij $w / s$ ligt, waarbij $s$ een schaal is die het rooster op de weights legt:
        """),
        L.formula(r"s = \frac{\max |w|}{q_{max}}, \qquad q = \mathrm{round}\left(\frac{w}{s}\right), \qquad \hat w = q \cdot s"),
        mo.md(r"""
        De weight die in grootte het grootst is, komt precies op $\pm q_{max}$ terecht, en elke andere hooguit een halve stap $s/2$ van waar hij was. Wat wordt opgeslagen, is $q$, in $b$ bits, en $s$; het model rekent met $\hat w = q \cdot s$.

        **Groepen.** Eén grote weight rekt het rooster op voor alle weights die zijn schaal delen, en de kleine worden 0. Daarom wordt elke rij van een matrix in groepen van bijvoorbeeld 32 weights geknipt, en krijgt elke groep een eigen schaal. Een schaal neemt 16 bits in, dus bij groepen van 32 voegt hij een halve bit per weight toe: 4 bits worden 4,5.

        **Meet het, neem het niet aan.** Een kleine fout per weight zegt weinig over het model: sommige weights tellen zwaarder dan andere. Zijn loss op tekst waarop het nooit trainde, ervoor en erna, zegt wat het kostte.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent |
        |---|---|
        | **quantization** | getallen, hier de weights van een model, opslaan als hele getallen van een paar bits met een schaal, zodat ze minder geheugen innemen |
        | **schaal** ($s$) | het getal waarmee de hele getallen van een groep worden vermenigvuldigd om zijn weights terug te geven: zijn grootste $\lvert w \rvert$ gedeeld door $q_{max}$ |
        | **groep** | de weights die één schaal delen, zoals 32 weights naast elkaar in een rij van een matrix |
        | $q_{max}$ | het grootste hele getal van $b$ bits met evenveel niveaus onder 0 als erboven: $2^{b-1} - 1$ |
        | `t.amax(dim=1, keepdim=True)` | het grootste getal van `t` langs as 1, bewaard als een as van lengte 1 |
        | `torch.round(t)` | elk getal van `t` afgerond op het dichtstbijzijnde hele getal |
        | `t.clamp(lo, hi)` | elk getal van `t` binnen `lo` en `hi` gehouden |

        Het plaatje laat één groep van 32 weights van het model van de course zien, uit de MLP van zijn eerste block: de ringen zijn de weights zoals ze zijn, de stippen waar ze in $b$ bits terechtkomen, en de dunne lijnen het rooster van niveaus. Schuif omhoog naar 8 bits en omlaag naar 2.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_bits_cell():
    _labels = {"title": "Eén groep van 32 weights, in minder bits", "x": "plek in de groep", "y": "waarde",
               "alt": "De 32 weights van één groep van het model van de course als ringen, en de waarden die ze in het gekozen aantal bits krijgen als stippen op een rooster van niveaus."}

    def _frame(b):
        _b = int(b)
        _n = C.group_numbers(_b)
        _say = (f"**{_b} bits:** {_n['levels']} niveaus, een stap van {C.number(_n['step'], 4, 'nl')}; "
                f"de grootste fout is {C.number(_n['error'], 4, 'nl')}, onder een halve stap.")
        return C.group_picture(_b, _labels), _say

    explore_bits = L.explore("bits", "Bits per weight", _frame, start=4, low=2, high=8, step=1)
    explore_bits
    return (explore_bits,)


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        Bij 8 bits zitten de stippen op de ringen: 255 niveaus, te dicht op elkaar om te tekenen. Bij 4 bits zie je de 15 niveaus, en is elke stip naar zijn dichtstbijzijnde lijn verschoven. Bij 2 bits zijn er drie niveaus, $-s$, 0 en $s$: 22 van de 32 weights worden 0, en alleen de 10 grootste houden hun teken.
        """),
        L.watch_out("De hele getallen besparen geheugen; ze maken het rekenen zelf niet goedkoper. Met weights in 4 bits zet de computer elke groep meestal eerst terug in getallen van 16 bits, $q \\cdot s$, om te vermenigvuldigen, in zijn snelle geheugen bij de cores. De winst zit in wat hij uit het trage geheugen leest."),
        mo.md(r"""
        ### Recept

        Om een model in minder bits op te slaan:

        1. Kies het aantal bits $b$ en de grootte van de groepen, zoals 4 bits en groepen van 32.
        2. Knip voor elke matrix van het model elke rij in groepen, en geef elke groep zijn schaal: zijn grootste $\lvert w \rvert$ gedeeld door $q_{max} = 2^{b-1} - 1$.
        3. Rond elke weight af op zijn hele getal $q = \mathrm{round}(w / s)$, van $-q_{max}$ tot $q_{max}$; sla $q$ op in $b$ bits en de schalen in 16.
        4. Meet de loss van het model op tekst waarop het nooit trainde, ervoor en erna.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Vier weights als één rij en één groep, in 4 bits, in PyTorch. Voer de cel uit.
    """)
    return


@app.cell
def example_group():
    _w = torch.tensor([[0.12, -0.5, 0.31, 0.02]])  # één groep
    _qmax = 2 ** (4 - 1) - 1  # 4 bits: -7 tot 7
    # de grootste |w| komt op 7
    _scale = _w.abs().amax(dim=1, keepdim=True) / _qmax
    _q = torch.round(_w / _scale).clamp(-_qmax, _qmax)
    print("scale:", round(_scale.item(), 4))
    print("q:", _q.tolist())
    print("back:", [round(v, 4) for v in (_q * _scale)[0].tolist()])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De schaal is $0{,}5 / 7 = 0{,}0714$. De weight die in grootte het grootst is, $-0{,}5$, wordt $-7$ en komt precies terug; $0{,}12$ wordt 2, wat $0{,}1429$ teruggeeft; en $0{,}02$ is minder dan een halve stap, dus die wordt 0. Opgeslagen is de groep vier hele getallen van 4 bits en één schaal van 16: 32 bits in plaats van 128.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_quantize():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. Een matrix in minder bits

        Maak `quantize(w, bits, group)` af: de weights van de matrix `w` zoals het model ermee rekent nadat elke weight is opgeslagen in `bits` bits, met een schaal per groep van `group` weights van een rij. De eerste regels knippen elke rij al in groepen: `groups` heeft de vorm (rijen, groepen per rij, `group`). `groups.abs().amax(dim=2, keepdim=True)` geeft de grootste $\lvert w \rvert$ van elke groep, `torch.round` rondt elk getal af, en `.clamp(-qmax, qmax)` houdt de hele getallen binnen hun bereik. Een groep met alleen nullen heeft een schaal van 0: geef hem een schaal van 1, dan blijven zijn weights 0.

        De check probeert een voorbeeld, een matrix van twee rijen, in 4 bits met groepen van 2; daarna één schaal per rij, 2 en 8 bits, een groep nullen, en de vorm van het resultaat.
        """),
        L.task("Maak `quantize` af: de schaal van elke groep, de hele getallen, en de weights terug in de vorm van `w`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_quantize():
    box_quantize = L.code_box("quantize", lines=16)
    box_quantize
    return (box_quantize,)


@app.cell(hide_code=True)
def check_quantize(box_quantize):
    result_quantize = L.check("quantize", box_quantize.value)
    result_quantize
    return (result_quantize,)


@app.cell(hide_code=True)
def hints_quantize():
    L.hints(
        "quantize",
        [
            "De schaal van elke groep is zijn grootste $\\lvert w \\rvert$ gedeeld door `qmax`; `amax(dim=2, keepdim=True)` houdt één getal per groep over, in een vorm waarmee je `groups` in één keer deelt.",
            "`q` is `groups / scale`, afgerond en begrensd op `-qmax` en `qmax`; waarmee het model rekent, is `q * scale`, teruggezet in de vorm `(rows, cols)`.",
        ],
        why="Elke groep krijgt de schaal die zijn grootste weight op $\\pm q_{max}$ zet; afronden op hele getallen is het opslaan in weinig bits, en `q * scale` is waarmee het model rekent.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(rf"""
    ### Wat je model doet in minder bits

    De knop slaat elke matrix van je model uit de capstone (of dat van de course, als je er geen hebt) op met je `quantize`, in 8, 4, 3 en 2 bits met groepen van {C.GROUP}, en in 4 bits met één schaal per rij. Voor elk meet hij de grootte en de loss op {C.number(C.TOKENS, 0, 'nl')} tokens van verhalen waarop het model nooit trainde, en hij laat het model een verhaal schrijven in 4 bits en in 2. Dat duurt op een laptop ongeveer een minuut.
    """)
    return


@app.cell(hide_code=True)
def demo_button_cell():
    demo_button = mo.ui.run_button(label="Sla mijn model op in minder bits")
    demo_button
    return (demo_button,)


@app.cell(hide_code=True)
def demo_quantize(box_quantize, demo_button, result_quantize):
    if result_quantize.state != "correct":
        _out = mo.md("*Maak eerst `quantize` goed in oefening 1: de demo slaat je model ermee op.*")
    elif not demo_button.value:
        _out = mo.md("*Druk op de knop om je model in minder bits op te slaan.*")
    else:
        _r = C.demo(box_quantize.value, lambda _b, _g: mo.output.replace(mo.md(f"**Bezig met meten:** {_b} bits …")))
        if _r == "run":
            _out = mo.md("*Het laden van je model stopt met een fout. Slagen je modules in `work/projects/llm/` nog voor hun checks? Verwijder een bestand dat je veranderde, en check zijn oefening opnieuw.*")
        elif _r == "quantize":
            _out = mo.md("*Je `quantize` stopt met een fout op een matrix van het model, of geeft weights terug met een andere vorm of die geen getallen zijn. Werkt hij voor een matrix van honderden rijen?*")
        else:
            _base = _r["rows"][0]["loss"]
            _name = lambda _x: "32 (zoals het is)" if _x["bits"] == 32 else str(_x["bits"])
            _grp = lambda _x: "" if _x["group"] is None else ("een rij" if _x["group"] == "row" else str(_x["group"]))
            _table = "| bits | groep | grootte | loss op de verhalen | verschil |\n|---|---|---|---|---|\n" + "\n".join(
                f"| {_name(_x)} | {_grp(_x)} | {C.number(_x['bytes'] / 1e6, 1, 'nl')}\u00a0MB | {C.number(_x['loss'], 2, 'nl')} | "
                f"{'' if _x['bits'] == 32 else ('+' if _x['loss'] >= _base else '−') + C.number(abs(_x['loss'] - _base), 2, 'nl')} |" for _x in _r["rows"])
            _out = mo.vstack([
                mo.md(f"**{'Je model' if _r['source'] == 'yours' else 'Het model van de course'}**, {C.number(_r['params'], 0, 'nl')} parameters:\n\n" + _table),
                mo.Html(C.demo_picture(_r["rows"], {"title": "Loss op de verhalen, groepen van 32", "x": "bits per weight", "y": "loss",
                                                    "full": "zoals het is", "alt": "De loss op de verhalen tegen het aantal bits per weight, met het model zoals het is als een gestreepte lijn."})),
                mo.md("**In 4 bits** schrijft het:\n\n> " + _r["stories"][4].replace("\n", " ") + "\n\n**In 2 bits:**\n\n> " + _r["stories"][2].replace("\n", " ")),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_memory():
    mo.md(r"""
    ### Oefening 2. Een open model op je laptop
    """)
    return


@app.cell(hide_code=True)
def choice_memory_cell():
    choice_memory = L.choice(
        "memory",
        question="Een open model van 7 miljard parameters neemt in BF16 14 GB in. Je slaat zijn weights op in 4 bits, met een schaal van 16 bits voor elke groep van 32 weights. Hoe groot is het dan ongeveer?",
        options={
            "quarter": "3,5 GB: een kwart van 14",
            "half": "7 GB: de helft van 14",
            "right": "3,9 GB: 4,5 bits per weight",
        },
        correct="right",
        why={
            "right": "Elke weight neemt 4 bits in, en elke groep van 32 voegt een schaal van 16 bits toe, een halve bit per weight: 7 miljard × 4,5 bits is 31,5 miljard bits, 3,9 GB.",
            "quarter": "4 bits is een kwart van 16, maar elke groep van 32 slaat ook zijn schaal van 16 bits op: een halve bit meer per weight. Kijk nog eens naar *Groepen* in het idee.",
            "half": "De helft zou 8 bits per weight zijn. In 4 bits, met een halve bit voor de schalen, is het 4,5 bits in plaats van 16.",
        },
    )
    choice_memory
    return (choice_memory,)


@app.cell(hide_code=True)
def text_exercise_faster():
    mo.md(r"""
    ### Oefening 3. Waarom het sneller schrijft
    """)
    return


@app.cell(hide_code=True)
def explain_faster():
    explain_faster_cell = L.explain(
        "faster",
        question="Op je laptop schrijft het model in 4 bits zijn tekst sneller dan hetzelfde model in BF16, hoewel elke vermenigvuldiging nog steeds in 16 bits gebeurt. Waarom?",
        model_answer="Eén token schrijven leest elke weight van het model één keer uit het geheugen, en doet er maar ongeveer twee bewerkingen mee: het is memory-bound, dus zijn tijd is ongeveer de bytes die het leest gedeeld door hoe snel het geheugen ze levert. In 4 bits, met een halve bit voor de schalen, is er ongeveer 4,5 / 16, iets meer dan een kwart, zoveel te lezen. Elke groep terugzetten in getallen van 16 bits gebeurt in het snelle geheugen bij de cores, waar het weinig kost. Zo kan het model tot ongeveer drieënhalf keer zo snel schrijven, voor de kleine prijs in loss die ik mat.",
        points=[
            "een token schrijven leest elke weight één keer: memory-bound",
            "minder bits betekent minder bytes om te lezen voor elk token",
            "de vermenigvuldigingen zelf zijn niet wat de tijd kost",
        ],
    )
    explain_faster_cell
    return (explain_faster_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Minder bits: FP32, FP16 en BF16](/course/open/sys-number-formats?lang=nl) sloeg elk getal op als een eigen float, met een eigen exponent; quantization slaat hele getallen op die per groep één schaal delen, en daarom kan het omlaag tot 4 bits. Het model is dat uit [Capstone: je eigen kleine LLM pretrainen](/course/open/llm-pretrain-capstone?lang=nl), en aan zijn loss op verhalen die het nooit zag, zie je wat minder bits kosten. Open modellen worden op ongeveer deze manier in 4 of 8 bits gedeeld, zodat ze op een laptop passen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_quantize, choice_memory, explain_faster_cell):
    L.wrap_up(result_quantize, choice_memory, explain_faster_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
