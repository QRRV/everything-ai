# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rnn-lstm. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rnn-lstm", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een recurrent neural network en backprop through time](/course/open/rnn-vanilla?lang=nl) en [Van score naar kans: de sigmoid](/course/open/ml-sigmoid?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_factor():
    quick_factor = L.choice(
        "quick-factor",
        question="In een recurrent neural network met een hidden state van één getal, $h_t = \\tanh(w h_{t-1} + x_t)$, waarmee vermenigvuldigt één stap terug de gradient?",
        options={
            "right": "$w (1 - h_t^2)$",
            "part": "$(1 - h_t^2) \\, h_{t-1}$",
            "weight": "$w$",
        },
        correct="right",
        why={
            "weight": "De gradient gaat ook door tanh, waarvan de helling bijna overal onder 1 ligt. Zie [Een recurrent neural network en backprop through time](/course/open/rnn-vanilla?lang=nl).",
            "part": "Die factor geeft het deel van stap $t$ voor $w$: daar vermenigvuldigt $w$ de $h_{t-1}$. Door naar $h_{t-1}$ komt de gradient langs $w$. Zie [Een recurrent neural network en backprop through time](/course/open/rnn-vanilla?lang=nl).",
            "right": "Een weight maal de helling van tanh, één keer per stap: over veel stappen verdwijnt dat product. Deze les verandert de weg die de gradient neemt.",
        },
    )
    quick_factor
    return (quick_factor,)


@app.cell(hide_code=True)
def quick_check_range():
    quick_range = L.choice(
        "quick-range",
        question="Welke getallen kan de sigmoid $\\sigma(z)$ geven?",
        options={
            "tanh": "Tussen −1 en 1",
            "right": "Tussen 0 en 1",
            "any": "Elk getal, net als $z$ zelf",
        },
        correct="right",
        why={
            "tanh": "Dat is tanh. De sigmoid maakt van een score een kans. Zie [Van score naar kans: de sigmoid](/course/open/ml-sigmoid?lang=nl).",
            "any": "De sigmoid perst elke $z$ in een getal tussen 0 en 1. Zie [Van score naar kans: de sigmoid](/course/open/ml-sigmoid?lang=nl).",
            "right": "Tussen 0 en 1: vermenigvuldig een getal ermee, en je houdt een deel van dat getal over. Dat doet een gate.",
        },
    )
    quick_range
    return (quick_range,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een gewoon recurrent neural network vergeet: de gradient van ver terug verdwijnt, dus het leert nauwelijks wat een vroeg woord met het eind moet doen. De **LSTM**, uit 1997, is precies voor dit probleem gemaakt. Voordat transformers het vanaf 2017 overnamen, vertaalden LSTM's teksten, herkenden ze spraak en voorspelden ze het volgende woord op het toetsenbord van telefoons; ze draaien nog steeds waar een model een lange stroom stap voor stap moet lezen. De **GRU** is een eenvoudiger versie die vaak ongeveer even goed werkt.
    """)
    return


@app.cell(hide_code=True)
def predict_keep():
    guess_keep = L.guess(
        "keep-100",
        question="Een LSTM houdt naast zijn hidden state een tweede vector bij. Bij elke stap wordt één getal ervan vermenigvuldigd met 0,99, en er komt niets bij. Hoeveel is er na 100 stappen van dat getal over?",
        options={
            "all": "Alles",
            "right": "Ongeveer een derde",
            "nothing": "Bijna niets, net als in een gewoon recurrent neural network",
        },
        reveal="**Ongeveer een derde:** $0{,}99^{100} \\approx 0{,}37$. Dat is veel: in een gewoon recurrent neural network kwam minder dan een duizendste van de gradient 30 stappen terug. De gradient terug langs deze vector wordt met dezelfde 0,99 per stap vermenigvuldigd, dus die reikt ook ver terug.",
    )
    guess_keep
    return (guess_keep,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een tweede geheugen.** Een LSTM neemt twee vectoren mee van stap naar stap: de hidden state $\mathbf{h}$, en de **cell state** $\mathbf{c}$. De cell state is zijn lange geheugen; de hidden state is wat hij er bij elke stap van laat zien.

        **Gates.** Een **gate** is een vector van getallen tussen 0 en 1, gemaakt met een sigmoid. Element voor element vermenigvuldigd, met $\odot$ zoals in [Backprop door hele layers met de hand](/course/open/nn-backprop-tensors?lang=nl), beslist hij hoeveel van elk getal doorgaat: 0 niets, 1 alles. Een LSTM heeft er drie:

        - de **forget gate** $\mathbf{f}_t$: hoeveel van de oude cell state blijft;
        - de **input gate** $\mathbf{i}_t$: hoeveel van een nieuwe **kandidaat** $\mathbf{g}_t$ erin gaat;
        - de **output gate** $\mathbf{o}_t$: hoeveel van de cell state de hidden state laat zien.

        **Eén stap.** Net als in een gewoon recurrent neural network gaan de input en de hidden state ervoor door weights: $\mathbf{z}_t = \mathbf{W}_x \mathbf{x}_t + \mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{b}$, nu met vier keer zoveel rijen. De vier delen ervan zijn, in de volgorde van PyTorch, $\mathbf{i}_t$, $\mathbf{f}_t$, $\mathbf{g}_t$ en $\mathbf{o}_t$: de gates door een sigmoid, de kandidaat door tanh. Dan:
        """),
        L.formula(r"[[hidden:\mathbf{c}_t]] = \mathbf{f}_t \odot [[hidden:\mathbf{c}_{t-1}]] + \mathbf{i}_t \odot \mathbf{g}_t"),
        L.formula(r"[[hidden:\mathbf{h}_t]] = \mathbf{o}_t \odot \tanh([[hidden:\mathbf{c}_t]])"),
        mo.md(r"""
        **Waarom de gradient het overleeft.** Terug van $\mathbf{c}_t$ naar $\mathbf{c}_{t-1}$ wordt de gradient vermenigvuldigd met $\mathbf{f}_t$, en dat is alles: geen weight-matrix en geen helling van tanh, die een gewoon recurrent neural network bij elke stap wel heeft. Waar de forget gate dicht bij 1 ligt, gaat de gradient bijna onveranderd terug langs de cell state, veel stappen lang. Tijdens het trainen leert het netwerk wanneer het moet bewaren, $\mathbf{f}_t$ dicht bij 1, en wanneer het moet vergeten, dicht bij 0.

        **De GRU** heeft geen cell state en twee gates: een **reset gate** $\mathbf{r}_t$, die beslist hoeveel van $\mathbf{h}_{t-1}$ in een nieuwe kandidaat $\mathbf{n}_t$ gaat, en een **update gate** $\mathbf{z}_t$, die de oude hidden state mengt met die kandidaat. Deze $\mathbf{z}_t$ is een gate, zoals PyTorch hem noemt, niet de $\mathbf{z}_t$ van de LSTM:
        """),
        L.formula(r"[[hidden:\mathbf{h}_t]] = (1 - \mathbf{z}_t) \odot \mathbf{n}_t + \mathbf{z}_t \odot [[hidden:\mathbf{h}_{t-1}]]"),
        mo.md(r"""
        Met drie delen in plaats van vier heeft een GRU een kwart minder weights. PyTorch heeft ze allebei: `nn.LSTM` en `nn.GRU` lezen een hele rij, `nn.LSTMCell` zet één stap.

        De figuur: hoeveel elke input de laatste hidden state nog verandert, tegen hoeveel stappen voor het eind hij gelezen werd, gemeten met PyTorch bij de beginweights en gemiddeld over 8 netwerken met een hidden state van 16 getallen. De y-as heeft een logaritmische schaal: gelijke stappen omlaag zijn gelijke factoren. PyTorch laat de forget gate rond 0,5 beginnen. De doorgetrokken lijn hoort bij dezelfde LSTM's, met 3 opgeteld bij de bias van hun forget gate, zodat die dicht bij 0,95 begint. Een grotere bias voor de forget gate aan het begin is een bekende truc: Jozefowicz, Zaremba en Sutskever (2015) raden hem aan.
        """),
        mo.Html(C.reach_picture({"title": "Hoe ver de gradient terugreikt", "x": "stappen terug vanaf het eind", "y": "deel dat het eind haalt",
                                 "rnn": "recurrent neural network", "lstm": "LSTM, forget gate 0,5", "keep": "LSTM, forget gate 0,95",
                                 "lang": L.lang,
                                 "alt": "Drie lijnen op een logaritmische schaal van 10 tot 10 tot de min 16, tegen 0 tot 59 stappen terug. Het gewone recurrent neural network, grijs en gestreept, daalt tot ongeveer 10 tot de min 16 na 59 stappen. De LSTM met zijn forget gate rond 0,5, gestreept, daalt tot ongeveer 10 tot de min 12. De LSTM met zijn forget gate dicht bij 0,95, doorgetrokken, blijft tussen ongeveer 0,08 en 1."})),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **LSTM** | long short-term memory: een recurrent neural network met een cell state en drie gates | `nn.LSTM` |
        | **cell state**, $\mathbf{c}_t$ | de tweede vector die een LSTM van stap naar stap meeneemt; hij verandert alleen via de gates | `c` |
        | **gate** | een vector van getallen tussen 0 en 1, uit een sigmoid, die element voor element beslist hoeveel er doorgaat | |
        | **forget gate**, **input gate**, **output gate**: $\mathbf{f}_t$, $\mathbf{i}_t$, $\mathbf{o}_t$ | hoeveel van de oude cell state blijft, hoeveel van de kandidaat erin gaat, en hoeveel van de cell state de hidden state laat zien | `f`, `i`, `o` |
        | **kandidaat**, $\mathbf{g}_t$ | wat de stap bij de cell state zou kunnen optellen, uit tanh | `g` |
        | **GRU** | gated recurrent unit: geen cell state, een reset gate $\mathbf{r}_t$ en een update gate $\mathbf{z}_t$ | `nn.GRU` |
        """),
        L.watch_out("Niet elke tekst schrijft de GRU zo. Sommige verwisselen $\\mathbf{z}_t$ en $1 - \\mathbf{z}_t$, zodat $\\mathbf{z}_t$ bij de nieuwe kandidaat hoort. Het idee is hetzelfde; kijk welke een paper of een library gebruikt voordat je getallen vergelijkt."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Eén stap van een LSTM:

    1. Reken $\mathbf{z}_t = \mathbf{W}_x \mathbf{x}_t + \mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{b}$ uit en splits hem in vier even grote delen.
    2. Input gate, forget gate en output gate: de sigmoid van hun delen; de kandidaat: de tanh van zijn deel.
    3. Nieuwe cell state: de forget gate maal de oude cell state, plus de input gate maal de kandidaat.
    4. Nieuwe hidden state: de output gate maal de tanh van de nieuwe cell state.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Eén stap van een LSTM met een cell state van één getal, vanaf stap 2 tot en met 4 van het recept: de vier delen van $\mathbf{z}_t$ zijn gegeven. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 3 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_step():
    import math

    _z = [2.0, 4.0, -0.5, 1.0]
    _c = 1.5
    # 2. de gates door een sigmoid, de kandidaat door tanh
    _i = 1 / (1 + math.exp(-_z[0]))
    _f = 1 / (1 + math.exp(-_z[1]))
    _g = math.tanh(_z[2])
    _o = 1 / (1 + math.exp(-_z[3]))
    print("i, f, g, o:", round(_i, 4), round(_f, 4), round(_g, 4), round(_o, 4))
    # 3. de nieuwe cell state
    _c_new = 0.0  # maak af: _f * _c + _i * _g
    # 4. de nieuwe hidden state
    _h_new = _o * math.tanh(_c_new)
    print("c:", round(_c_new, 4), " h:", round(_h_new, 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Zoals hij is, wordt de cell state 0, en de hidden state ook: alles is vergeten. Afgemaakt houdt de forget gate, 0,982, bijna alles van de oude 1,5, en laat de input gate, 0,8808, het grootste deel van de kandidaat, −0,4621, erin: de cell state wordt 1,066, en de hidden state $0{,}7311 \times \tanh(1{,}066) \approx 0{,}576$. De kandidaat is negatief, dus deze stap haalt een deel van de cell state weg: van 1,5 naar 1,066.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_step():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. Eén stap van een LSTM

        `lstm_step(x, h, c, W_x, W_h, b)` zet één stap van een LSTM, met de input `x`, de hidden state `h` en de cell state `c` van de stap ervoor, en geeft de nieuwe hidden state en de nieuwe cell state terug. `W_x` heeft shape (4 × hidden, inputs), `W_h` shape (4 × hidden, hidden), en `b` shape (4 × hidden,). De eerste regels maken arrays, rekenen `z` uit, en halen de input gate en de forget gate uit hun delen ervan. Vervang elke `...`: de kandidaat, de output gate, de nieuwe cell state en de nieuwe hidden state. De check roept `lstm_step([1.0], [0.5], [2.0], [[1.0], [2.0], [0.5], [-1.0]], [[0.5], [1.0], [1.0], [2.0]], [0.0, 1.0, 0.0, 0.5])` aan, wat ongeveer `([0.6147], [2.5334])` moet geven, en probeert een hidden state van 2 getallen, weights van 0, en gehele getallen.
        """),
        L.task("Laat `lstm_step` de nieuwe hidden state en de nieuwe cell state van één LSTM-stap geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_step():
    box_step = L.code_box("lstm-step", lines=18)
    box_step
    return (box_step,)


@app.cell(hide_code=True)
def check_step(box_step):
    result_step = L.check("lstm-step", box_step.value)
    result_step
    return (result_step,)


@app.cell(hide_code=True)
def hints_step():
    L.hints(
        "lstm-step",
        [
            "De kandidaat en de output gate komen uit het derde en het vierde deel van `z`, zoals `i` en `f` uit het eerste en het tweede komen. Dan stap 3 en 4 van het recept, met `*` voor $\\odot$.",
            "`g` gaat door `np.tanh`, `o` door `sigmoid`. De nieuwe hidden state gebruikt de nieuwe cell state.",
        ],
        why="`g = np.tanh(z[2 * n:3 * n])`, `o = sigmoid(z[3 * n:4 * n])`, `c_new = f * c + i * g` en `h_new = o * np.tanh(c_new)`. Voor het voorbeeld is $\\mathbf{z}$ gelijk aan (1,25; 3,5; 1; 0,5), dus is de forget gate 0,9707 en houdt hij het meeste van de oude 2.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Jouw stap en die van PyTorch

    De cel hieronder maakt de `nn.LSTMCell` van PyTorch voor inputs van 3 getallen en een hidden state van 2, geeft je `lstm_step` dezelfde weights, en vergelijkt de nieuwe hidden states. PyTorch houdt twee bias-vectoren bij, één voor de input en één voor de hidden state; opgeteld zijn ze je `b`.
    """)
    return


@app.cell(hide_code=True)
def demo_torch(box_step, result_step):
    _both = C.against_torch(box_step.value) if result_step.state == "correct" else None
    mo.md(
        f"Je `lstm_step` geeft de hidden state `{_both[0]}`, en `nn.LSTMCell` geeft `{_both[1]}`: dezelfde stap, in dezelfde volgorde van de gates."
        if _both else "Als je `lstm_step` goed is, wordt hij hier vergeleken met de `nn.LSTMCell` van PyTorch."
    )
    return


@app.cell(hide_code=True)
def text_exercise_forget():
    mo.md(r"""
    ### Oefening 2. Een forget gate van 0
    """)
    return


@app.cell(hide_code=True)
def choice_forget_cell():
    choice_forget = L.choice(
        "forget-zero",
        question="In een getrainde LSTM is bij stap $t$, voor één getal van de cell state, de forget gate bijna 0 en de input gate bijna 1. Wat gebeurt er met dat getal?",
        options={
            "add": "Het houdt zijn oude waarde en telt de kandidaat erbij op",
            "zero": "Het wordt 0, en blijft vanaf dan 0",
            "right": "Zijn oude waarde wordt vervangen door de kandidaat",
        },
        correct="right",
        why={
            "add": "De forget gate vermenigvuldigt de oude waarde: bijna 0 betekent dat er bijna niets van blijft.",
            "zero": "De input gate van bijna 1 laat de kandidaat erin, en latere stappen kunnen er weer iets bij optellen.",
            "right": "$0 \\cdot c_{t-1} + 1 \\cdot g_t = g_t$: de oude waarde is weg, en er gaat bijna geen gradient langs de cell state terug voorbij deze stap. Het netwerk heeft geleerd hier te vergeten, bijvoorbeeld aan het eind van een zin.",
        },
    )
    choice_forget
    return (choice_forget,)


@app.cell(hide_code=True)
def text_exercise_gru():
    mo.md(r"""
    ### Oefening 3. Een GRU die kopieert
    """)
    return


@app.cell(hide_code=True)
def choice_gru_cell():
    choice_gru = L.choice(
        "gru-copy",
        question="Een GRU, geschreven zoals in *Het idee*, heeft een update gate $\\mathbf{z}_t$ van 1 voor elk getal bij elke stap. Wat doet zijn hidden state?",
        options={
            "right": "Hij blijft zoals hij was: elke stap kopieert $\\mathbf{h}_{t-1}$",
            "new": "Hij vergeet alles: elke stap neemt alleen de kandidaat",
            "mean": "Hij wordt het gemiddelde van de oude hidden state en de kandidaat",
        },
        correct="right",
        why={
            "new": "Met $\\mathbf{z}_t = 1$ krijgt de kandidaat $1 - 1 = 0$. Lees de formule in *Het idee* nog eens.",
            "mean": "Dat zou $\\mathbf{z}_t = 0{,}5$ zijn. Met 1 gaat al het gewicht naar één van de twee.",
            "right": "$(1 - 1) \\odot \\mathbf{n}_t + 1 \\odot \\mathbf{h}_{t-1} = \\mathbf{h}_{t-1}$, en de gradient gaat onveranderd terug. Zo bewaart een GRU iets lang, net als een LSTM met een forget gate van 1.",
        },
    )
    choice_gru
    return (choice_gru,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De stap van [Recurrentie: een netwerk met geheugen](/course/open/rnn-memory-idea?lang=nl), met gates gemaakt door de sigmoid van [Van score naar kans: de sigmoid](/course/open/ml-sigmoid?lang=nl). Langs de cell state worden de factoren van [Een recurrent neural network en backprop through time](/course/open/rnn-vanilla?lang=nl) forget gates, en die kunnen dicht bij 1 blijven. Transformers nemen een andere weg om ver terug te reiken: elke positie kijkt direct naar elke andere positie, zonder stap voor stap te gaan.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_forget, choice_gru, result_step):
    L.wrap_up(result_step, choice_forget, choice_gru)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
