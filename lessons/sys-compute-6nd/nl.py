# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-compute-6nd. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-compute-6nd", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Parameters en FLOPs per token tellen](/course/open/gpt-count?lang=nl) en [Geheugen of compute: wat remt je GPU af?](/course/open/sys-memory-roofline?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_six():
    quick_six = L.choice(
        "quick-six",
        question="Een token lezen kost een GPT ongeveer 2 FLOPs per parameter, en erop trainen ongeveer 6. Waar komen de andere 4 vandaan?",
        options={
            "right": "Uit de backward pass, die ongeveer twee keer de forward pass kost",
            "adam": "Uit AdamW, die voor elk token 4 FLOPs per parameter doet",
            "twice": "Uit elk token drie keer lezen",
        },
        correct="right",
        why={
            "right": "Voor elke parameter doet de backward pass per token twee vermenigvuldigingen: één voor de gradient van die parameter, één om de gradient door te geven aan de laag ervoor. De stap van de optimizer komt één keer per mini-batch, niet per token.",
            "adam": "AdamW werkt elke parameter één keer per stap bij, voor een hele mini-batch tokens: een paar FLOPs per parameter per stap, die nauwelijks tellen. Zie [Parameters en FLOPs per token tellen](/course/open/gpt-count?lang=nl).",
            "twice": "Trainen leest elk token één keer, forward, en gaat dan één keer terug, backward. Zie [Parameters en FLOPs per token tellen](/course/open/gpt-count?lang=nl).",
        },
    )
    quick_six
    return (quick_six,)


@app.cell(hide_code=True)
def quick_check_peak():
    quick_peak = L.choice(
        "quick-peak",
        question="Een GPU doet volgens zijn specificaties 100 TFLOP/s. Een trainingsstap kost $6 \\times 10^{12}$ FLOPs. Hoe lang duurt de stap minstens?",
        options={
            "ms": "0,6 milliseconde",
            "right": "0,06 seconde",
            "s": "6 seconden",
        },
        correct="right",
        why={
            "ms": "100 TFLOP/s is $10^{14}$ FLOPs per seconde: $6 \\times 10^{12} / 10^{14} = 0{,}06$ seconde. Zie [Geheugen of compute: wat remt je GPU af?](/course/open/sys-memory-roofline?lang=nl).",
            "s": "Een TFLOP is $10^{12}$ FLOPs, dus 100 TFLOP/s is $10^{14}$ per seconde, niet $10^{12}$. Zie [Geheugen of compute: wat remt je GPU af?](/course/open/sys-memory-roofline?lang=nl).",
            "right": "$6 \\times 10^{12} / 10^{14} = 0{,}06$ seconde, op de volle piek. Echte code doet er langer over: hij haalt maar een deel van de piek.",
        },
    )
    quick_peak
    return (quick_peak,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Vóór een trainingsrun wil je weten hoe lang hij duurt en wat hij kost: een uur op je laptop, een nacht op een gehuurde GPU, of meer dan je kunt betalen. Grote labs plannen zo runs van maanden op duizenden GPU's, en jij kunt je eigen runs op dezelfde manier plannen. Eén vuistregel geeft het antwoord voordat je iets draait, uit drie getallen: de grootte van het model, het aantal tokens, en de snelheid van je GPU's.
    """)
    return


@app.cell(hide_code=True)
def predict_gpt3():
    guess_gpt3 = L.guess(
        "gpt3-one-gpu",
        question="GPT-3 heeft 175 miljard parameters en trainde op 300 miljard tokens. Trainen op één token kost ongeveer 6 FLOPs per parameter. Hoe lang zou één A100-GPU erover doen, op zijn volle piek van 312 TFLOP/s in BF16?",
        options={
            "months": "Ongeveer 3 maanden",
            "years": "Ongeveer 3 jaar",
            "right": "Ongeveer 30 jaar",
        },
        reveal="**Ongeveer 30 jaar.** $6 \\times 175 \\times 10^9 \\times 300 \\times 10^9 \\approx 3{,}2 \\times 10^{23}$ FLOPs, met $312 \\times 10^{12}$ per seconde: $10^9$ seconden, 32 jaar. Dat is op de volle piek, die geen run haalt. Het artikel over GPT-3 meldt $3{,}14 \\times 10^{23}$ FLOPs voor zijn training, gedaan op een groot cluster van GPU's.",
    )
    guess_gpt3
    return (guess_gpt3,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Stap 1: de hele run, 6ND.** Trainen op één token kost ongeveer $6N$ FLOPs voor een model van $N$ parameters. Een run die op $D$ tokens traint, kost dan
        """),
        L.formula(r"\text{FLOPs} \approx 6\,N\,D"),
        mo.md(r"""
        $D$ telt elk token waarop het model traint: 20 epochs over een boek van 1 miljoen tokens is $D = 20$ miljoen. De regel laat het deel van attention weg dat met het venster groeit: queries met keys vergelijken, en de values mengen. Bij het trainen voegt dat deel ongeveer $n/(12d)$ van $6N$ toe, voor vensters van $n$ tokens en vectoren van $d$ getallen: ongeveer 5,6% voor je GPT van het boek, met $n = 256$ en $d = 384$, en 1,4% voor GPT-3, met $n = 2.048$ en $d = 12.288$.

        Het plaatje: wat trainen op één token kost, in eenheden van $N$ FLOPs: de forward pass, $2N$, de backward pass, $4N$, en het smalle stuk aan het eind, attention.
        """),
        mo.Html(C.token_picture({"title": "Trainen op één token", "x": "FLOPs, in eenheden van N", "forward": "forward", "backward": "backward",
                                 "row": "{name}: +{pct}%", "book": "je GPT van het boek", "gpt3": "GPT-3", "lang": L.lang,
                                 "alt": "Twee balken, elk 6N lang: de forward pass, 2N, en de backward pass, 4N. Attention voegt er 5,6% aan toe voor je GPT van het boek, en 1,4% voor GPT-3."})),
        mo.md(r"""
        **Stap 2: van FLOPs naar tijd.** De specificaties van een GPU geven zijn piek, per getalformaat: een A100 doet 312 TFLOP/s in BF16 en 19,5 TFLOP/s in FP32 op zijn gewone cores; een H100 SXM doet 989 TFLOP/s in BF16. Een echte run haalt maar een deel van de piek: grote runs ongeveer 30 tot 50%, kleine modellen vaak veel minder. De tijd is het aantal FLOPs gedeeld door wat alle GPU's samen echt doen in een seconde:
        """),
        L.formula(r"\text{tijd} = \frac{6\,N\,D}{\text{piek} \times \text{deel} \times \text{GPU's}}"),
        mo.md(r"""
        **Stap 3: GPU-uren en de rekening.** De tijd maal het aantal GPU's geeft **GPU-uren**, en GPU-uren maal de prijs per GPU-uur geven de rekening. Twee keer zoveel GPU's maken de run ongeveer twee keer zo kort, voor ongeveer evenveel GPU-uren.

        **Stap 4: terugrekenen.** Omgedraaid zegt dezelfde regel wat er in een budget past, $D = \text{FLOPs} / (6N)$: met hetzelfde budget traint een model dat twee keer zo groot is op half zoveel tokens. En uit de GPU-uren die een lab meldt, volgt welk deel van de piek hun run haalde.
        """),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **6ND-vuistregel** | een model van $N$ parameters trainen op $D$ tokens kost ongeveer $6ND$ FLOPs | `6 * n_params * n_tokens` |
        | **GPU-uur** | één GPU die één uur wordt gebruikt; 8 GPU's 10 uur lang is 80 GPU-uur | `hours * n_gpus` |
        | $D$ | het aantal tokens waarop een model traint, epochs meegeteld | `n_tokens` |
        """),
        L.watch_out("Neem de piek van het getalformaat waarin je traint, en zonder sparsity: sommige specificaties geven een twee keer zo hoge piek die alleen geldt voor matrices met twee nullen in elk groepje van vier getallen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo schat je een trainingsrun:

    1. Tel de parameters $N$, en de tokens $D$ waarop het model traint: tokens per stap maal stappen, of tokens in de data maal epochs.
    2. Vermenigvuldig: $6ND$ FLOPs.
    3. Deel door de piek van je GPU in je getalformaat, door het deel ervan dat je verwacht te halen, en door het aantal GPU's: dat is de tijd in seconden.
    4. Vermenigvuldig de tijd in uren met het aantal GPU's voor de GPU-uren, en die met de prijs van één GPU-uur voor de rekening.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Nieuw is het terugrekenen vanuit een verslag. Llama 2 70B heeft 70 miljard parameters en trainde op 2 biljoen tokens; het artikel erover meldt 1.720.320 GPU-uren op A100-GPU's. Hoeveel van hun piek haalden die GPU's?
    """)
    return


@app.cell
def example_llama():
    _n = 70e9                  # parameters van Llama 2 70B
    _d = 2e12                  # tokens waarop het trainde
    _peak = 312e12             # een A100 in BF16, FLOPs per seconde
    _reported = 1_720_320      # GPU-uren, in het artikel over Llama 2

    _flops = 6 * _n * _d
    _at_peak = _flops / _peak / 3600
    print("FLOPs:", _flops)
    print("GPU-uren op de volle piek:", round(_at_peak))
    print("deel van de piek gehaald:", round(_at_peak / _reported, 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Op de volle piek zouden $8{,}4 \times 10^{23}$ FLOPs 747.863 GPU-uren kosten; de run kostte er 1.720.320. Over de hele run haalden de GPU's ongeveer 43% van hun piek. De rest ging op aan dingen zoals bewerkingen die op het geheugen wachten, GPU's die op elkaar wachten, en al het andere dat een run doet naast matrices vermenigvuldigen. De volgende les meet dat deel op je eigen machine.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_hours():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. Hoe lang?

        Schrijf `train_hours(n_params, n_tokens, peak, utilization, n_gpus)`: de tijd in uren om een model van `n_params` parameters te trainen op `n_tokens` tokens, op `n_gpus` GPU's met elk een piek van `peak` FLOPs per seconde, waarvan ze het deel `utilization` halen, zoals `0.4`. De check roept `train_hours(1e9, 20e9, 989e12, 0.4, 8)` aan, een model van een miljard parameters op 8 H100-GPU's, wat ongeveer 10,53 moet geven, en probeert andere runs.
        """),
        L.task("Schrijf `train_hours(n_params, n_tokens, peak, utilization, n_gpus)`: de tijd van de run in uren, met de 6ND-vuistregel."),
    ])
    return


@app.cell(hide_code=True)
def exercise_hours():
    box_hours = L.code_box("train-hours", lines=6)
    box_hours
    return (box_hours,)


@app.cell(hide_code=True)
def check_hours(box_hours):
    result_hours = L.check("train-hours", box_hours.value)
    result_hours
    return (result_hours,)


@app.cell(hide_code=True)
def hints_hours():
    L.hints(
        "train-hours",
        [
            "Stap 2 en 3 van het recept: de FLOPs van de hele run, gedeeld door wat alle GPU's samen echt doen in een seconde. Dat geeft seconden; jij wilt uren.",
        ],
        why="$6 \\times 10^9 \\times 20 \\times 10^9 = 1{,}2 \\times 10^{20}$ FLOPs, met $989 \\times 10^{12} \\times 0{,}4 \\times 8 \\approx 3{,}2 \\times 10^{15}$ per seconde: 37.917 seconden, 10,53 uur.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Van een boek tot 70 miljard parameters

    De cel hieronder laat je `train_hours` vier runs uitrekenen, allemaal op A100-GPU's op 40% van hun piek in BF16: je GPT van het boek, 10,8 miljoen parameters op 20 epochs over een boek van ongeveer een miljoen tokens, op één GPU; een model van 124 miljoen parameters, de grootte van GPT-2 small, op 10 miljard tokens op 8 GPU's; en modellen van 7 en 70 miljard parameters op 2 biljoen tokens, de groottes en de aantallen tokens van Llama 2, op 1.024 en 2.048 GPU's. Hij werkt zodra oefening 1 goed is.
    """)
    return


@app.cell(hide_code=True)
def demo_runs(box_hours, result_hours):
    if result_hours.state != "correct":
        _out = mo.md("*Maak eerst oefening 1 af: deze cel gebruikt je `train_hours`.*")
    else:
        _rows = C.demo(box_hours.value)
        if _rows == "error":
            _out = mo.md("*Je `train_hours` stopte met een fout bij een van deze runs.*")
        else:
            _names = {"book": "je GPT, het boek, 1 GPU", "small": "124 miljoen, 8 GPU's", "7b": "7 miljard, 1.024 GPU's", "70b": "70 miljard, 2.048 GPU's"}
            _gh = lambda _g: C.number(_g, 3 if _g < 1 else 0, L.lang)
            _table = "\n".join(f"| {_names[_k]} | {C.duration(_h, L.lang)} | {_gh(_g)} |" for _k, _h, _g in _rows)
            _out = mo.md(f"""
    | run | tijd | GPU-uren |
    |---|---|---|
    {_table}

    Van seconden tot meer dan een maand, met dezelfde regel. Voor het model van 70 miljard parameters schat je functie ongeveer 1,9 miljoen GPU-uren; het artikel over Llama 2 meldt er 1,7 miljoen, omdat zijn GPU's iets meer dan 40% haalden. Een model zo klein als je GPT van het boek haalt veel minder dan 40%: zijn matrices zijn te klein om een grote GPU bezig te houden, en veel van zijn bewerkingen wachten op het geheugen. De volgende les meet hoeveel hij wel haalt.
    """)
    _out
    return


@app.cell(hide_code=True)
def text_exercise_budget():
    mo.md(r"""
    ### Oefening 2. Een budget
    """)
    return


@app.cell(hide_code=True)
def choice_budget_cell():
    choice_budget = L.choice(
        "budget",
        question="Je mag 64 H100-GPU's gebruiken, met 989 TFLOP/s in BF16, een week lang, en je verwacht 40% van hun piek te halen. Je traint een model van 7 miljard parameters. Op ongeveer hoeveel tokens kan het trainen?",
        options={
            "hours": "Ongeveer 100 miljoen",
            "right": "Ongeveer 365 miljard",
            "forward": "Ongeveer 1,1 biljoen",
        },
        correct="right",
        why={
            "hours": "Dat krijg je als je de week in uren telt, terwijl de piek in FLOPs per seconde is. Een week heeft 604.800 seconden.",
            "forward": "Dat krijg je met 2 FLOPs per parameter per token, alleen de forward pass. Trainen kost er 6.",
            "right": "$64 \\times 604.800 \\times 989 \\times 10^{12} \\times 0{,}4 \\approx 1{,}5 \\times 10^{22}$ FLOPs, gedeeld door $6 \\times 7 \\times 10^9$: ongeveer $3{,}6 \\times 10^{11}$ tokens.",
        },
    )
    choice_budget
    return (choice_budget,)


@app.cell(hide_code=True)
def text_exercise_optimistic():
    mo.md(r"""
    ### Oefening 3. Langer dan gepland
    """)
    return


@app.cell(hide_code=True)
def explain_optimistic():
    explain_optimistic_cell = L.explain(
        "too-optimistic",
        question="Met de 6ND-vuistregel en 40% van de piek schatte je een run op 10 dagen. Hij duurt 14 dagen. Noem twee redenen waarom de schatting te kort kan uitvallen, en hoe je dat vóór de run had kunnen ontdekken.",
        model_answer="**De run haalde minder van de piek dan 40%:** bewerkingen die op het geheugen wachten, kleine matrices, GPU's die op elkaar of op data wachten, evaluaties en checkpoints onderweg. **De run doet meer FLOPs dan 6ND:** attention op lange vensters komt erbovenop, en delen van de forward pass opnieuw berekenen, om geheugen te besparen, voegt tot een derde toe. Herstarts na een crash of een slecht stuk training kosten ook tijd. **Om het vooraf te ontdekken:** draai een paar honderd stappen op de echte hardware, meet de tokens per seconde, en deel de tokens die je nodig hebt daardoor; de volgende les doet dat.",
        points=[
            "je noemt een lager deel van de piek dan gepland, met een reden zoals wachten op het geheugen, op andere GPU's of op data",
            "je noemt een tweede reden, zoals attention op lange vensters, opnieuw berekenen of herstarts",
            "je zegt dat je eerst een korte run op de echte hardware meet",
        ],
    )
    explain_optimistic_cell
    return (explain_optimistic_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De 6 FLOPs per parameter per token komen uit [Parameters en FLOPs per token tellen](/course/open/gpt-count?lang=nl), de piek van een GPU uit [Geheugen of compute: wat remt je GPU af?](/course/open/sys-memory-roofline?lang=nl), en BF16 uit [Mixed-precision training](/course/open/sys-mixed-precision?lang=nl). Samen met [Waar het geheugen heen gaat](/course/open/sys-memory-accounting?lang=nl) kun je nu vóór een run zeggen of hij past en hoe lang hij duurt. De volgende les, *Profiling: tokens per seconde en MFU*, meet welk deel van de piek je eigen runs halen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_budget, explain_optimistic_cell, result_hours):
    L.wrap_up(result_hours, choice_budget, explain_optimistic_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
