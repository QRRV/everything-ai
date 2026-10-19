# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-number-formats. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-number-formats", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_before():
    mo.md(r"""
    **Ook van buiten de course:** binaire getallen: een **bit** is een 0 of een 1, $n$ bits geven $2^n$ verschillende patronen, en binaire cijfers achter de komma tellen halven, kwarten, achtsten enzovoort, dus 1,1 binair is 1,5. Is dat weggezakt, lees dan eerst [Binair talstelsel](https://nl.wikipedia.org/wiki/Binair_talstelsel).
    """)
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Floating point en de log-sum-exp-truc](/course/open/num-floats-stability?lang=nl) en [Geheugen of compute: wat remt je GPU af?](/course/open/sys-memory-roofline?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_overflow():
    quick_overflow = L.choice(
        "quick-overflow",
        question="Een float bewaart een getal als een paar significante cijfers keer een macht van 2. Wat gebeurt er met een uitkomst die een grotere macht van 2 nodig heeft dan waarvoor het formaat ruimte heeft?",
        options={
            "wrap": "Hij springt om naar een negatief getal",
            "right": "Hij wordt `inf`",
            "largest": "Hij wordt afgerond op het grootste getal dat het formaat kan bewaren",
        },
        correct="right",
        why={
            "wrap": "Gehele getallen springen in sommige talen om; floats niet. Zie [Floating point en de log-sum-exp-truc](/course/open/num-floats-stability?lang=nl).",
            "largest": "Een float heeft er geen ruimte voor, en wordt `inf`: een overflow. Zie [Floating point en de log-sum-exp-truc](/course/open/num-floats-stability?lang=nl).",
            "right": "Dat is een overflow. Met minder bits voor de macht gebeurt dat veel eerder, zoals deze les laat zien.",
        },
    )
    quick_overflow
    return (quick_overflow,)


@app.cell(hide_code=True)
def quick_check_halve():
    quick_halve = L.choice(
        "quick-halve",
        question="Een bewerking is memory-bound. Je halveert de bytes van elk getal dat ze leest en schrijft. Wat gebeurt er volgens de roofline-schatting met haar tijd?",
        options={
            "same": "Die blijft gelijk: de FLOPs veranderen niet",
            "right": "Die halveert ongeveer",
            "quarter": "Die daalt tot ongeveer een kwart",
        },
        correct="right",
        why={
            "same": "Een memory-bound bewerking wacht op haar bytes, niet op haar FLOPs. Zie [Geheugen of compute: wat remt je GPU af?](/course/open/sys-memory-roofline?lang=nl).",
            "quarter": "Haar tijd is haar bytes gedeeld door de bandbreedte: de helft van de bytes, de helft van de tijd. Zie [Geheugen of compute: wat remt je GPU af?](/course/open/sys-memory-roofline?lang=nl).",
            "right": "Haar tijd is bytes gedeeld door bandbreedte, zolang ze memory-bound blijft. Dat is de eerste reden om minder bits te gebruiken.",
        },
    )
    quick_halve
    return (quick_halve,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    PyTorch bewaart elke float in 32 bits, tenzij je om minder vraagt. De helft van de bits is de helft van het geheugen, de helft van de bytes om te verplaatsen, en op moderne GPU's een andere eenheid om mee te rekenen: de **tensor cores**, die kleine blokken van matrices in 16-bits formaten veel sneller vermenigvuldigen. Een NVIDIA A100 doet 19,5 biljoen FP32-bewerkingen per seconde op zijn gewone cores, en 312 biljoen in BF16 op zijn tensor cores. Daarom doen grote modellen het meeste rekenwerk van hun training in 16 bits. Maar minder bits laten minder ruimte: voor grote en kleine getallen, of voor cijfers. Welke bits een formaat inlevert, bepaalt wat er misgaat.
    """)
    return


@app.cell(hide_code=True)
def predict_store():
    guess_store = L.guess(
        "seventy-thousand",
        question="Je bewaart het getal 70.000 in twee formaten van 16 bits, FP16 en BF16. Wat komt eruit?",
        options={
            "both": "70.000 in allebei",
            "right": "`inf` in FP16, ongeveer 70.000 in BF16",
            "swap": "Ongeveer 70.000 in FP16, `inf` in BF16",
        },
        reveal="**`inf` in FP16, 70.144 in BF16.** FP16 heeft 5 bits voor de macht van 2, en zijn grootste getal is 65.504: 70.000 loopt over. BF16 heeft 8 bits voor de macht, evenveel als FP32, dus 70.000 past, maar voor de cijfers heeft het maar 7 bits: rond 70.000 liggen zijn getallen 512 uit elkaar, en 70.000 wordt afgerond op het dichtstbijzijnde, 70.144.",
    )
    guess_store
    return (guess_store,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Stap 1: drie delen.** Een floating-point-getal is een **teken**, een macht van 2 en een aantal significante cijfers, binair geschreven: $\pm\, 1{,}b_1 b_2 \ldots b_m \times 2^{e}$. De bits van de **exponent** bewaren $e$ en bepalen het **bereik**: hoe groot en hoe klein een getal kan worden. De bits van de **mantisse** bewaren de $m$ binaire cijfers achter de komma, en bepalen de **numerieke precisie**, hoe fijn de stappen zijn: tussen twee machten van 2, $2^e$ en $2^{e+1}$, maken $m$ mantissebits $2^m$ gelijke stappen. Dat is iets anders dan de precision van een classifier. Elk getal wordt afgerond op het dichtstbijzijnde getal dat het formaat kan bewaren.

        | Formaat | Bits | Teken, exponent, mantisse | Grootste getal | Stap bij 1 |
        |---|---|---|---|---|
        | **FP32** | 32 | 1, 8, 23 | $3{,}4 \times 10^{38}$ | $1{,}2 \times 10^{-7}$ |
        | **TF32** | 19, binnen tensor cores | 1, 8, 10 | $3{,}4 \times 10^{38}$ | $9{,}8 \times 10^{-4}$ |
        | **BF16** | 16 | 1, 8, 7 | $3{,}4 \times 10^{38}$ | $7{,}8 \times 10^{-3}$ |
        | **FP16** | 16 | 1, 5, 10 | 65.504 | $9{,}8 \times 10^{-4}$ |

        **Stap 2: twee manieren om te halveren.** **FP16** houdt meer cijfers, ongeveer 3 significante, maar heeft maar 5 exponentbits: boven 65.504 loopt een getal over, onder ongeveer $6 \times 10^{-5}$ houdt het minder cijfers, en onder ongeveer $3 \times 10^{-8}$ wordt het 0. Kleine gradients kunnen zo verdwijnen; wie in FP16 traint, moet daar iets tegen doen. **BF16**, brain floating point, houdt de 8 exponentbits van FP32 en levert cijfers in: het heeft het bereik van FP32 en ongeveer 2 significante cijfers. Het loopt zelden over, en het is tegenwoordig het gebruikelijke 16-bits formaat om in te trainen, op GPU's die het ondersteunen.

        **Stap 3: TF32.** Sinds 2020 kunnen de tensor cores van NVIDIA-GPU's ook FP32-matrices vermenigvuldigen met maar 10 mantissebits: **TF32**. De tensors blijven FP32 in het geheugen; alleen de vermenigvuldiging binnen de tensor core rondt af op minder cijfers. PyTorch staat het toe met `torch.set_float32_matmul_precision("high")`.

        **Stap 4: gehele getallen.** **INT8** en **INT4** bewaren gehele getallen, 256 en 16 verschillende waarden, met één schaalfactor voor een groep getallen. Ze bewaren de weights van een getraind model in een kwart of een achtste van het geheugen van FP32; een model daarin omzetten is quantization, in een latere les.

        **Stap 5: de grootte van een stap.** Bij een getal $x$ zijn de stappen van een formaat
        """),
        L.formula(r"\text{spacing}(x) = 2^{\lfloor \log_2 |x| \rfloor - m}"),
        mo.md(r"""
        met $m$ mantissebits, voor elke $x$ die niet 0 is, niet te groot, en niet zo klein dat hij minder cijfers houdt. Tel je een getal kleiner dan een halve stap bij $x$ op, dan verandert $x$ helemaal niet: hij rondt weer af op zichzelf. Omlaag vanaf een macht van 2 is die grens half zo groot, want net onder $2^e$ gelden de stappen van de macht eronder, die half zo groot zijn.

        Het plaatje: de bits van elk formaat, teken, exponent en mantisse.
        """),
        mo.Html(C.bits_picture({"title": "Waar de bits heen gaan", "bits": "bits", "sign": "teken", "exp": "exponent", "man": "mantisse", "note_TF32": ", binnen tensor cores",
                                "alt": "Vier rijen vakjes, één per bit. FP32: 1 tekenbit, 8 exponentbits, 23 mantissebits. TF32: 1, 8 en 10, 19 bits, binnen tensor cores. BF16: 1, 8 en 7, 16 bits. FP16: 1, 5 en 10, 16 bits."})),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **exponent**, **mantisse** | de bits van een float die zijn macht van 2 bewaren, en zijn significante cijfers |
        | **bereik**, **numerieke precisie** | hoe groot en hoe klein de getallen van een formaat kunnen worden, en hoe fijn zijn stappen zijn; niet de precision van een classifier |
        | **FP32**, **FP16**, **BF16** | floating-point-formaten van 32 en 16 bits; BF16 heeft het bereik van FP32 met minder cijfers |
        | **TF32** | het formaat waarin tensor cores FP32-matrices vermenigvuldigen, met 10 mantissebits |
        | **INT8**, **INT4** | formaten voor gehele getallen van 8 en 4 bits, met een schaalfactor, om weights in te bewaren |
        | **tensor core** | een eenheid in een NVIDIA-GPU die kleine blokken van matrices in één keer vermenigvuldigt, het snelst in 16 bits of minder |
        | `torch.float16`, `torch.bfloat16` | de dtypes van FP16 en BF16 in PyTorch; `x.to(torch.bfloat16)` zet om |
        """),
        L.watch_out("Oudere GPU's, en de CPU van veel laptops, hebben geen snelle BF16: daar kan het zelfs trager zijn dan FP32. Kijk wat je hardware ondersteunt voordat je overstapt."),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Getalformaten in PyTorch zijn nieuw: drie getallen bewaard in FP32, FP16 en BF16.
    """)
    return


@app.cell
def example_formats():
    import torch

    x = torch.tensor([70000.0, 1e-8, 1.0 + 1 / 256])
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        print(dtype, x.to(dtype).tolist())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    FP32 houdt alle drie; alleen $10^{-8}$ wordt afgerond, in de laatste cijfers. FP16 maakt van 70.000 `inf` en van $10^{-8}$ 0: te groot en te klein voor 5 exponentbits. BF16 houdt allebei, afgerond, maar maakt van $1 + 1/256$ gewoon 1: zijn stappen bij 1 zijn $1/128$, dus $1 + 1/256$ ligt precies midden tussen 1 en $1 + 1/128$. Ligt een getal precies in het midden, dan wordt het afgerond op het getal van die twee waarvan de laatste mantissebit 0 is, hier 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_spacing():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Hoe groot is een stap?

        Schrijf `spacing(x, mantissa_bits)`: de stap tussen de getallen die een formaat met `mantissa_bits` mantissebits kan bewaren, in de macht van 2 waarin `abs(x)` ligt, zoals in stap 5, voor elke `x` die niet 0 is. `math.log2` en `math.floor` zitten in de module `math` van Python. De check roept `spacing(70000.0, 7)` aan, wat 512 moet geven, de stappen van BF16 uit de gok, en probeert andere getallen en formaten.
        """),
        L.task("Schrijf `spacing(x, mantissa_bits)`: de stap tussen de getallen van een formaat met zoveel mantissebits, in de macht van 2 waarin `abs(x)` ligt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_spacing():
    box_spacing = L.code_box("spacing", lines=6)
    box_spacing
    return (box_spacing,)


@app.cell(hide_code=True)
def check_spacing(box_spacing):
    result_spacing = L.check("spacing", box_spacing.value)
    result_spacing
    return (result_spacing,)


@app.cell(hide_code=True)
def hints_spacing():
    L.hints(
        "spacing",
        [
            "Eerst de macht van 2 waarin `x` ligt: `math.floor(math.log2(abs(x)))`. Die macht, van $2^p$ tot $2^{p+1}$, wordt verdeeld in `2 ** mantissa_bits` gelijke stappen.",
        ],
        why="```python\nimport math\n\n\ndef spacing(x, mantissa_bits):\n    power = math.floor(math.log2(abs(x)))\n    return 2.0 ** (power - mantissa_bits)\n```\n\nTussen $2^p$ en $2^{p+1}$ ligt een stuk met lengte $2^p$, en de mantisse verdeelt dat in $2^m$ stappen van $2^{p-m}$. Het teken doet er niet toe, vandaar `abs`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_format():
    mo.md(r"""
    ### Oefening 2. Welk formaat?
    """)
    return


@app.cell(hide_code=True)
def choice_format_cell():
    choice_format = L.choice(
        "format",
        question="Je traint een groot model op een recente GPU die BF16 ondersteunt, en je wilt de matrixvermenigvuldigingen in minder bits doen dan FP32. Sommige activations worden een paar honderdduizend, en sommige gradients zijn zo klein als $10^{-9}$. In welk formaat doe je ze?",
        options={
            "fp16": "FP16, omdat het meer cijfers houdt",
            "right": "BF16, omdat het het bereik van FP32 heeft",
            "int8": "INT8, omdat het het minste geheugen kost",
        },
        correct="right",
        why={
            "fp16": "Zijn extra cijfers helpen niet als een paar honderdduizend boven 65.504 overloopt en $10^{-9}$ 0 wordt.",
            "int8": "INT8 heeft 256 gehele getallen en één schaalfactor: rek je die uit tot een paar honderdduizend, dan liggen de stappen duizenden uit elkaar en wordt $10^{-9}$ 0. Het is om de weights van een getraind model te bewaren, niet om mee te trainen.",
            "right": "Met 8 exponentbits passen zowel een paar honderdduizend als $10^{-9}$. Zijn stappen zijn grof, en de volgende les vangt dat op door sommige getallen in FP32 te houden.",
        },
    )
    choice_format
    return (choice_format,)


@app.cell(hide_code=True)
def text_exercise_lost():
    mo.md(r"""
    ### Oefening 3. Een verloren update
    """)
    return


@app.cell(hide_code=True)
def explain_lost():
    explain_lost_cell = L.explain(
        "lost-update",
        question="Een weight van 1,0 staat in BF16. Elke trainingsstap telt er 0,001 bij op, 1000 stappen achter elkaar. Wat is de weight na die stappen, en waarom? Wat zou je veranderen?",
        model_answer="**Nog steeds 1,0.** Van 1 tot 2 zijn de stappen van BF16 $2^{0-7} = 1/128 \\approx 0{,}0078$. De update, 0,001, is kleiner dan een halve stap, 0,0039, dus $1{,}0 + 0{,}001$ rondt weer af op 1,0, elke stap opnieuw. De weight had op 2,0 moeten uitkomen, en komt nooit van zijn plek. **Veranderen:** houd een kopie van de weights in FP32, waarvan de stappen bij 1 $1{,}2 \\times 10^{-7}$ zijn, en pas de updates op die kopie toe; gebruik BF16 alleen voor het rekenen. Dat is mixed-precision training, de volgende les.",
        points=[
            "je zegt dat de weight 1,0 blijft",
            "je vergelijkt de update met de stap van BF16 bij 1, ongeveer 0,0078, en de helft daarvan, ongeveer 0,004",
            "je stelt voor de weights in een formaat met fijnere stappen te houden, zoals FP32, en de updates daarop toe te passen",
        ],
    )
    explain_lost_cell
    return (explain_lost_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Overflow, underflow en afronden komen uit [Floating point en de log-sum-exp-truc](/course/open/num-floats-stability?lang=nl), de bytes die minder bits besparen uit [Geheugen of compute: wat remt je GPU af?](/course/open/sys-memory-roofline?lang=nl). De volgende les, *Mixed-precision training*, rekent in BF16 waar dat veilig is en houdt de weights in FP32, zodat updates zoals die uit oefening 3 niet verloren gaan.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_format, explain_lost_cell, result_spacing):
    L.wrap_up(result_spacing, choice_format, explain_lost_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
