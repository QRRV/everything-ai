# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les num-floats-stability. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("num-floats-stability", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Cross-entropy-loss voor veel klassen](/course/open/ml-cross-entropy?lang=nl) en [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_loss():
    quick_loss = L.choice(
        "quick-loss",
        question="Softmax geeft de goede klasse een kans van 0,9. Door training wordt de logit van die klasse iets groter. Wat gebeurt er met de cross-entropy-loss?",
        options={
            "right": "Die daalt: richting 0, maar nooit eronder",
            "up": "Die stijgt: een grotere logit geeft een grotere loss",
            "same": "Die blijft gelijk: 0,9 is al bijna goed",
        },
        correct="right",
        why={
            "up": "Een grotere logit voor de goede klasse geeft haar een grotere kans, en min de log van een grotere kans is kleiner. Zie [Cross-entropy-loss voor veel klassen](/course/open/ml-cross-entropy?lang=nl).",
            "same": "De loss is min de log van de kans, ongeveer 0,105 bij 0,9; die wordt pas 0 bij een kans van 1. Zie [Cross-entropy-loss voor veel klassen](/course/open/ml-cross-entropy?lang=nl).",
            "right": "De kans van de goede klasse stijgt, dus min haar log daalt, richting 0, en 0 wordt die pas bij een kans van 1.",
        },
    )
    quick_loss
    return (quick_loss,)


@app.cell(hide_code=True)
def quick_check_log():
    quick_log = L.choice(
        "quick-log",
        question="Wat is $\\log(e^2 \\cdot e^5)$?",
        options={
            "times": "10: de exponenten vermenigvuldigd",
            "right": "7: de exponenten opgeteld",
            "power": "Ongeveer 1097: e tot de macht 7",
        },
        correct="right",
        why={
            "times": "Machten van $e$ vermenigvuldigen telt hun exponenten op: $e^2 \\cdot e^5 = e^7$. Zie [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl).",
            "power": "$e^2 \\cdot e^5$ is inderdaad $e^7$, ongeveer 1097, maar de log maakt de $e$-macht ongedaan. Zie [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl).",
            "right": "$e^2 \\cdot e^5 = e^7$, en de log maakt de $e$-macht ongedaan: 7. Precies die stap heb je nodig voor de truc in deze les.",
        },
    )
    quick_log
    return (quick_log,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een computer bewaart elk getal in een vast aantal bits. Meestal merk je daar niets van, maar een model rekent met heel grote en heel kleine getallen: $e$ tot de macht een logit, kansen die bijna 0 zijn, sommen van miljoenen termen. Dan kan een tussenstap te groot zijn om te bewaren, terwijl het antwoord dat je zoekt een gewoon getal is, en wordt de loss van een trainingsrun `nan`. GPU's gebruiken vaak nog minder bits, dus daar gebeurt het eerder.

    **Een voorbeeld uit het dagelijks leven.** Een rekenmachine toont tien cijfers. Vraag hem om $10^{100} / 10^{99}$ door eerst $10^{100}$ in te typen, en hij geeft een foutmelding, terwijl het antwoord 10 is.
    """)
    return


@app.cell(hide_code=True)
def predict_point_three():
    guess_sum = L.guess(
        "point-three",
        question="Wat denk je dat `0.1 + 0.2 == 0.3` in Python geeft?",
        options={
            "true": "`True`: 0,1 plus 0,2 is 0,3",
            "error": "Een foutmelding: `==` werkt niet op floats",
            "right": "`False`: de twee kanten verschillen een beetje",
        },
        reveal="**`False`.** `0.1 + 0.2` geeft `0.30000000000000004`. Een computer bewaart getallen in machten van 2, en 0,1 heeft geen exacte vorm in machten van 2, net zoals 1/3 geen exacte vorm heeft als decimaal getal: 0,3333… houdt ergens op. Dus 0,1, 0,2 en 0,3 worden elk afgerond, en de afronding van `0.1 + 0.2` komt op een ander getal uit dan de afronding van 0,3.",
    )
    guess_sum
    return (guess_sum,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **floating-point** getal, of float, wordt bewaard als een getal van ongeveer 16 significante cijfers keer een macht van 2. De gewone float van NumPy, float64, gebruikt daar 64 bits voor. Door de macht passen in dezelfde 64 bits zowel $10^{-300}$ als $10^{300}$; door de 16 cijfers wordt bijna elk getal afgerond. De afstand tussen een float en de volgende groeit met het getal mee: naast 1 is die ongeveer $2{,}2 \cdot 10^{-16}$, naast $10^{16}$ is die 2, dus `1e16 + 1 == 1e16` is `True`.

        De grootste float64 is ongeveer $1{,}8 \cdot 10^{308}$. Een uitkomst daarboven wordt `inf`, oneindig groot: **overflow**. $e^{709}$ past nog, $e^{710}$ niet. Een uitkomst die dichter bij 0 ligt dan ongeveer $10^{-308}$ houdt minder cijfers over, en onder ongeveer $5 \cdot 10^{-324}$ wordt die 0: **underflow**. $e^{-746}$ is 0, en NumPy waarschuwt je daar niet voor, zoals bij overflow wel. GPU's rekenen vaak met minder bits: in float32 loopt $e^{89}$ al over, in float16 $e^{12}$.

        De loss van een classifier of een language model neemt $e$ tot de macht van elke logit, telt die op en neemt de log: de **log-sum-exp**. Met logits van duizend lopen de $e$-machten over, terwijl het antwoord, ongeveer duizend, makkelijk past. De truc uit [Veel klassen: softmax](/course/open/ml-softmax?lang=nl) helpt ook hier: haal eerst de grootste waarde $m$ eraf, en tel hem na de log weer op.
        """),
        L.formula(r"\mathrm{LSE}(\mathbf{z}) = \log \sum_k e^{z_k} = m + \log \sum_k e^{z_k - m}, \quad m = \max_k z_k"),
        mo.md(r"""
        Dat mag omdat $e^{z_k} = e^m \cdot e^{z_k - m}$: de som is $e^m$ keer de som van de verschoven machten, de log van een product is de som van de logs, en $\log(e^m) = m$. Elke exponent $z_k - m$ is 0 of lager, dus geen enkele macht loopt over; en de grootste macht is $e^0 = 1$, dus de som is minstens 1 en je neemt nooit de log van 0.

        De cross-entropy-loss uit [Cross-entropy-loss voor veel klassen](/course/open/ml-cross-entropy?lang=nl) is min de log van de softmax-kans van de goede klasse $c$. Uitgeschreven is dat de log-sum-exp min de logit van de goede klasse:
        """),
        L.formula(r"[[loss:\mathcal{L}]] = -\log \frac{e^{z_c}}{\sum_k e^{z_k}} = \mathrm{LSE}(\mathbf{z}) - z_c"),
        mo.md(r"""
        Reken je eerst de softmax uit en daarna de log, dan kan een heel kleine kans door underflow 0 worden, en de log van 0 is min oneindig. Met deze formule schrijf je de kans nooit op, en blijft de loss een gewoon getal. Daarom krijgt de cross-entropy-loss in libraries meestal de logits, niet de kansen.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **floating point**, float | | een getal bewaard als een vast aantal significante cijfers, ongeveer 16 bij float64, keer een macht van 2 | `0.1`, `1e16` |
        | **overflow** | | een uitkomst die te groot is voor een float; die wordt `inf` | `np.exp(710)` |
        | **underflow** | | een uitkomst die te dicht bij 0 ligt voor een float; die wordt 0 | `np.exp(-746)` |
        | **log-sum-exp**, $\mathrm{LSE}(\mathbf{z})$ | log-sum-exp van z | de log van de som van $e$ tot de macht van elke waarde, uitgerekend als $m + \log \sum_k e^{z_k - m}$ | `m + np.log(np.sum(np.exp(z - m)))` |
        | $m$ | m | hier: de grootste waarde van $\mathbf{z}$ | `z.max()` |
        """),
        L.watch_out("Een float is niet fout, maar afgerond. Vergelijk twee uitkomsten nooit met `==`: `0.1 + 0.2 == 0.3` is `False`. Vergelijk met een kleine marge: `abs(a - b) < 1e-9`, waarbij `abs` een getal zonder zijn teken geeft. En een uitkomst die `inf` of `nan` is, verspreidt zich: elke som, elk product en elk gemiddelde ermee is ook `inf` of `nan`, dus één overflow vroeg in een berekening bederft alles erna."),
        mo.md(r"""
        De figuur: voor getallen van 1 tot $10^{16}$ de afstand tot de volgende float64. Op deze twee logschalen is dat een rechte lijn: de afstand groeit gelijk op met het getal (van dichtbij in piepkleine trapjes, want hij verandert alleen bij elke macht van 2), dus een float houdt altijd ongeveer 16 cijfers, waar de komma ook staat.
        """),
        mo.Html(C.gap_picture({"title": "De afstand tot de volgende float", "x": "het getal", "y": "afstand tot de volgende float64",
                               "at_one": "bij 1: 2,2 · 10⁻¹⁶", "at_big": "bij 10¹⁶: 2",
                               "alt": "Een lijn op twee logschalen, van de afstand 2,2 keer 10 tot de macht min 16 bij het getal 1, recht omhoog naar de afstand 2 bij het getal 10 tot de macht 16."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om de log-sum-exp uit te rekenen zonder overflow:

    1. Zoek de grootste waarde: $m = \max_k z_k$.
    2. Haal die van elke waarde af: $z_k - m$, zodat elke exponent 0 of lager is.
    3. Neem $e$ tot de macht van elk, tel ze op, en neem de log.
    4. Tel $m$ er weer bij. Voor de cross-entropy-loss haal je daarna de logit van de goede klasse eraf: $\mathrm{LSE}(\mathbf{z}) - z_c$.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Drie logits van ongeveer duizend, eerst met de gewone formule, dan met het recept. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 4 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar dat zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_lse():
    import numpy as np

    _z = np.array([1000.0, 1001.0, 999.0])
    # de gewone formule: e tot de macht 1000 loopt over
    print(np.log(np.sum(np.exp(_z))))
    # 1. de grootste waarde
    _m = _z.max()
    # 2. haal die eraf: elke exponent is 0 of lager
    _shifted = _z - _m
    print(_shifted)
    # 3. de e-machten, hun som, en de log daarvan
    _log_sum = np.log(np.sum(np.exp(_shifted)))
    # 4. tel m er weer bij. De loss van klasse 0 is dat min haar logit, _z[0]: print die ook
    print(_m + _log_sum)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De gewone formule geeft `inf`, en NumPy waarschuwt voor de overflow, in een regel met `RuntimeWarning` boven de uitkomsten. Verschoven zijn de logits −1, 0 en −2; hun machten, ongeveer 0,37, 1 en 0,14, tellen op tot ongeveer 1,50, met als log ongeveer 0,408. Met 1001 er weer bij is de log-sum-exp ongeveer 1001,408: net boven de grootste logit, zoals een log-sum-exp altijd is. Met stap 4 af is de loss van klasse 0 ongeveer 1001,408 − 1000 = 1,408: klasse 0 heeft de op een na grootste logit, en softmax geeft haar $e^{-1{,}408}$, ongeveer 0,24.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_underflow():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Een heel klein getal
    """)
    return


@app.cell(hide_code=True)
def choice_underflow_cell():
    choice_underflow = L.choice(
        "underflow",
        question="Wat geeft `np.exp(-800)`?",
        options={
            "tiny": "Een piepklein positief getal: floats gaan tot ongeveer 10⁻³⁰⁸",
            "right": "`0.0`: het antwoord ligt te dicht bij 0 voor een float",
            "minus": "`-inf`: de exponent is heel negatief",
        },
        correct="right",
        why={
            "tiny": "$e^{-800}$ is ongeveer $10^{-347}$, ver onder zelfs de kleinste float, ongeveer $5 \\cdot 10^{-324}$. Al $e^{-746}$ is 0.",
            "minus": "$e$ tot welke macht ook is groter dan 0, nooit negatief: een heel negatieve exponent geeft een getal dicht bij 0, geen negatief getal.",
            "right": "$e^{-800}$ is ongeveer $10^{-347}$: underflow. Al $e^{-746}$ is 0.",
        },
    )
    choice_underflow
    return (choice_underflow,)


@app.cell(hide_code=True)
def text_exercise_lse():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Log-sum-exp

        Schrijf `logsumexp(z)`: de log van de som van $e$ tot de macht van elke waarde van de array `z`, met het recept, zodat hij voor alle waarden werkt. De check roept `logsumexp(np.array([1, 2, 3]))` aan, en dat moet ongeveer `3.4076` geven; hij probeert ook waarden van duizend en van min duizend.
        """),
        L.task("Laat `logsumexp` de log-sum-exp van `z` geven, zonder overflow."),
    ])
    return


@app.cell(hide_code=True)
def exercise_lse():
    box_lse = L.code_box("logsumexp", lines=6)
    box_lse
    return (box_lse,)


@app.cell(hide_code=True)
def check_lse(box_lse):
    result_lse = L.check("logsumexp", box_lse.value)
    result_lse
    return (result_lse,)


@app.cell(hide_code=True)
def hints_lse():
    L.hints(
        "logsumexp",
        [
            "Stap 1 en 2 van het recept: `m = z.max()`, en `z - m` voor alle waarden tegelijk.",
            "Stap 3 en 4: `np.exp`, `np.sum` en `np.log` van de verschoven waarden, en dan `m` er weer bij.",
        ],
        why="Voor `[1, 2, 3]`: $m$ = 3, de verschoven waarden zijn −2, −1 en 0, hun machten tellen op tot ongeveer 1,503, en $3 + \\log(1{,}503) \\approx 3{,}4076$, hetzelfde als wat de gewone formule voor deze kleine waarden geeft.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_ce():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De loss rechtstreeks uit de logits

        Schrijf `ce_from_logits(z, c)`: de cross-entropy-loss van de logits `z` als de goede klasse `c` is, als de log-sum-exp min de logit van de goede klasse, zonder overflow. De check roept `ce_from_logits(np.array([2, 1, 0]), 0)` aan, en dat moet ongeveer `0.4076` geven; hij probeert ook andere klassen en logits van duizend en meer.
        """),
        L.task("Laat `ce_from_logits` de cross-entropy-loss van `z` voor klasse `c` geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ce():
    box_ce = L.code_box("ce-from-logits", lines=6)
    box_ce
    return (box_ce,)


@app.cell(hide_code=True)
def check_ce(box_ce):
    result_ce = L.check("ce-from-logits", box_ce.value)
    result_ce
    return (result_ce,)


@app.cell(hide_code=True)
def hints_ce():
    L.hints(
        "ce-from-logits",
        [
            "Eerst de log-sum-exp van `z`, met het recept, zoals in oefening 2: je mag die regels naar dit vak kopiëren.",
            "Haal daarna de logit van de goede klasse eraf: `z[c]`.",
        ],
        why="Voor `[2, 1, 0]` is de log-sum-exp ongeveer 2,4076 en de logit van klasse 0 is 2: de loss is ongeveer 0,4076, min de log van de softmax-kans 0,665.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De grootste logit eraf halen, uit [Veel klassen: softmax](/course/open/ml-softmax?lang=nl), houdt ook de log-sum-exp uit de overflow, omdat $\log(e^m \cdot S) = m + \log S$, zoals in [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl). Daarmee komt de cross-entropy-loss uit [Cross-entropy-loss voor veel klassen](/course/open/ml-cross-entropy?lang=nl) rechtstreeks uit de logits, zonder kans die 0 kan worden. Je komt de truc weer tegen in *Attention weights uit inproducten*, waar softmax grote scores krijgt. In de volgende les, *Hoe zeker is een getal? Betrouwbaarheidsintervallen en de bootstrap*, zoek je uit hoeveel een getal dat je mat anders had kunnen uitvallen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_underflow, result_lse, result_ce):
    L.wrap_up(choice_underflow, result_lse, result_ce)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
