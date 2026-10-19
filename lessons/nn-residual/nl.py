# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nn-residual. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow derivative

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-residual", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Vanishing en exploding gradients](/course/open/nn-vanishing-exploding?lang=nl) en [Differentieerregels](/course/open/math-derivative-rules?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_product():
    quick_product = L.choice(
        "quick-product",
        question="Op zijn weg terug door 20 layers wordt de gradient vermenigvuldigd met één factor per layer. Elke factor is ongeveer 0,5. Hoe groot is ongeveer wat bij de eerste layer aankomt, als hij bovenaan 1 was?",
        options={
            "half": "Ongeveer 0,5: elke layer halveert dezelfde gradient één keer",
            "ten": "Ongeveer 0,1: twintig keer halveren maakt hem tien keer kleiner",
            "right": "Ongeveer 0,000001: twintig keer halveren na elkaar, één keer per layer",
        },
        correct="right",
        why={
            "half": "De factoren worden vermenigvuldigd: na twee layers is het 0,25, na drie 0,125. Zie [Vanishing en exploding gradients](/course/open/nn-vanishing-exploding?lang=nl).",
            "ten": "Drie of vier keer halveren maakt er al een tiende van. Twintig keer maakt er $0{,}5^{20}$ van. Zie [Vanishing en exploding gradients](/course/open/nn-vanishing-exploding?lang=nl).",
            "right": "$0{,}5^{20} \\approx 0{,}00000095$: de gradient verdwijnt. Vandaag zie je een manier om elke factor dicht bij 1 te houden.",
        },
    )
    quick_product
    return (quick_product,)


@app.cell(hide_code=True)
def quick_check_sum_rule():
    quick_sum_rule = L.choice(
        "quick-sum-rule",
        question="$y = x + x^2$. Wat is de afgeleide van $y$ naar $x$?",
        options={
            "square": "$2x$: de losse $x$ is een constante",
            "right": "$1 + 2x$: de afgeleiden van de twee termen opgeteld",
            "product": "$x \\cdot 2x$: de afgeleiden van de twee termen vermenigvuldigd",
        },
        correct="right",
        why={
            "square": "De losse $x$ is geen constante: hij verandert met $x$, met afgeleide 1. Zie [Differentieerregels](/course/open/math-derivative-rules?lang=nl).",
            "product": "Bij een som tel je de afgeleiden op. Vermenigvuldigen hoort bij een keten van functies. Zie [Differentieerregels](/course/open/math-derivative-rules?lang=nl).",
            "right": "De afgeleide van een som is de som van de afgeleiden: 1 voor $x$, en $2x$ voor $x^2$. Die 1 is de kern van de les van vandaag.",
        },
    )
    quick_sum_rule
    return (quick_sum_rule,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een diep netwerk zou minstens moeten kunnen wat een minder diep netwerk kan: de extra layers kunnen hun input gewoon doorgeven. Toch trainden diepere gewone netwerken in de praktijk vaak slechter, ook op hun eigen training data, en ook met batch normalization. In [Vanishing en exploding gradients](/course/open/nn-vanishing-exploding?lang=nl) zag je een reden: op zijn weg terug wordt de gradient vermenigvuldigd met één factor per layer, en veel factoren onder 1 laten bijna niets over voor de eerste layers. Residual connections veranderden dat in 2015: netwerken van meer dan 100 layers trainden goed, en elke transformer gebruikt ze.
    """)
    return


@app.cell(hide_code=True)
def predict_gradient():
    guess_gradient = L.guess(
        "first-block",
        question="Een MLP voor de cijfers heeft 10 blocks van twee layers van 64 neuronen, met de eigen beginweights van PyTorch. Aan het begin van het trainen is de gradient die bij het eerste block aankomt meer dan een miljoen keer kleiner dan in het laatste block. Nu telt elk block zijn input op bij zijn output; verder verandert er niets. Hoe groot is de gradient in het eerste block nu volgens jou, vergeleken met het laatste?",
        options={
            "million": "Nog steeds meer dan een miljoen keer kleiner",
            "thousand": "Ongeveer duizend keer kleiner",
            "right": "Ongeveer even groot als in het laatste block",
        },
        reveal="**Ongeveer even groot.** In het laatste block is hij 0,00051, in het eerste 0,00042. Zonder het optellen daalde hij van 0,000038 naar 0,000000000025. Het optellen geeft de gradient een directe weg terug door elk block.",
    )
    guess_gradient
    return (guess_gradient,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **residual connection** telt de input van een block op bij zijn output. Het block berekent een $f(x)$, zoals een layer, een ReLU en nog een layer, en geeft
        """),
        L.formula(r"y = x + f(x)"),
        mo.md(r"""
        $f$ hoeft alleen een verandering van $x$ te leren, en met $f(x) = 0$ geeft het block $x$ ongewijzigd door: een extra block kan altijd niets doen. Op de terugweg is de afgeleide van een som de som van de afgeleiden:
        """),
        L.formula(r"\frac{\partial y}{\partial x} = 1 + \frac{\partial f}{\partial x}"),
        mo.md(r"""
        Door $n$ blocks wordt de gradient vermenigvuldigd met $n$ factoren $1 + f'$, niet met $n$ factoren $f'$. Waar $f'$ klein is, zoals bij een verzadigde sigmoid, is elke factor nog steeds ongeveer 1: de gradient verdwijnt niet meer. Als $x$ een vector is, zoals in een layer, komt de gradient voor $y$ ongewijzigd terug bij $x$, plus wat er door $f$ terugkomt. In code is het block één regel: `x + f(x)`. Dat werkt alleen als er evenveel getallen uit $f$ komen als erin gaan.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **residual connection** | | de input van een block optellen bij zijn output: $y = x + f(x)$ | `x + f(x)` |
        | $f(x)$ | f van x | wat het block uit zijn input berekent, hier layer, ReLU, layer | `self.lin2(torch.relu(self.lin1(x)))` |
        """),
        L.watch_out("Niet het residu uit [Hoe ver zit één voorspelling ernaast?](/course/open/ml-residuals?lang=nl): daar is het het label min de voorspelling; hier zegt het woord dat het block alleen leert wat het erbij moet optellen. En een residual connection houdt de getallen niet tegen als ze groeien: elk block telt iets op bij $x$. In het netwerk uit de figuur wordt de loss met een learning rate van 0,3 na 20 stappen nog steeds `nan`. De normalisatie uit de volgende les helpt daarbij."),
        mo.md(r"""
        De figuur: de gemiddelde grootte van de gradient van de weights van de eerste layer in elk block, aan het begin van het trainen, voor een MLP met de eigen beginweights van PyTorch, een input-layer, 10 blocks van twee layers van 64 neuronen en 10 logits, na één backward pass op 256 trainingscijfers. Zonder residual connections geeft elk block $\mathrm{ReLU}(f(h))$, en daalt de gradient van 0,000038 in het laatste block tot 0,000000000025 in het eerste. Met residual connections geeft elk block $h + f(h)$, en ligt de gradient in elk block tussen ongeveer 0,0004 en 0,0007. Eén epoch getraind met een learning rate van 0,1 blijft het netwerk zonder residual connections op 2,30, de loss van gokken; met residual connections leest het 838 van de 1.000 testcijfers goed.
        """),
        mo.Html(C.grad_picture({"title": "De gradient in elk van 10 blocks, aan het begin", "x": "block", "y": "gemiddelde grootte van de gradient",
                                "plain": "zonder residual connections", "res": "met residual connections",
                                "alt": "Twee lijnen op een logaritmische schaal over 10 blocks. Zonder residual connections, gestippeld, stijgt de gradient van ongeveer 0,00000000003 in block 1 naar ongeveer 0,00004 in block 10. Met residual connections blijft hij tussen ongeveer 0,0004 en 0,0007."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo geef je een block een residual connection:

    1. Laat $f$ evenveel getallen teruggeven als erin gaan: de laatste layer van $f$ heeft evenveel outputs als het block inputs heeft.
    2. Tel de input erbij op: $y = x + f(x)$, met niets na het optellen.
    3. Stapel de blocks: de input van het eerste heeft een directe weg naar het laatste, heen en terug.
    4. Volg de gradient terug, van het laatste block naar het eerste, en vermenigvuldig voor elk block met $1 + f'$: 1 voor de directe weg, plus wat er door $f$ terugkomt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De keten uit [Vanishing en exploding gradients](/course/open/nn-vanishing-exploding?lang=nl): 10 sigmoid-layers, elk met weight 1, en de input 0,5, waar ongeveer 0,00000035 terugkwam. Nu geeft elke layer $h + \mathrm{sigmoid}(w h)$. Het commentaar met `#` zegt welke stap van het recept elk deel is. De vergelijking aan het eind is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar dat zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_residual_chain():
    import numpy as np

    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    _w = 1.0
    _h = 0.5
    _zs = []
    # 1. tot en met 3. de forward pass door 10 gestapelde layers: elk telt sigmoid(z) op bij zijn input, en we bewaren elke z
    for _k in range(10):
        _z = _w * _h
        _zs.append(_z)
        _h = _h + _sigmoid(_z)
    print(np.round(_zs, 3))
    # 4. terug van de laatste layer naar de eerste: keer 1 plus de weight keer de helling van de sigmoid
    _g = 1.0
    for _z in reversed(_zs):
        _g = _g * (1 + _w * _sigmoid(_z) * (1 - _sigmoid(_z)))
        print(_g)
    # vergelijk: print hoeveel keer meer er terugkomt dan de 0.00000035 zonder residual connections,
    #    _g / 0.00000035, en voer opnieuw uit; zet daarna _w op 4.0 en voer nog een keer uit
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    $h$ groeit in elke layer met hooguit 1, dus de $z$'s groeien ook: eerst 0,5, dan 1,122 en 1,877, en zo verder tot 8,645. Ver van 0 is de sigmoid verzadigd en zijn helling piepklein, maar elke factor is 1 plus die helling, dus hij komt nooit onder 1. Op de terugweg is de gradient na de laatste layer 1,0002, en hij groeit tot 1,79 bij de input: ongeveer 5 miljoen keer meer dan zonder residual connections. Met `_w = 4.0` komt er ongeveer 1,44 terug. Je vergelijking deelt nog door de 0,00000035 van weight 1; met weight 4 liet de gewone keten maar ongeveer 0,00000000005 door, dus nu komt er ongeveer 26 miljard keer meer terug.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_harmless():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Een extra block
    """)
    return


@app.cell(hide_code=True)
def choice_harmless_cell():
    choice_harmless = L.choice(
        "why-harmless",
        question="Je voegt 10 blocks met residual connections toe aan een netwerk dat al goed traint. Waarom maakt dat het zelden slechter?",
        options={
            "skipped": "Tijdens het trainen slaat PyTorch blocks die niet helpen over",
            "right": "Elk block kan $f(x) \\approx 0$ leren, en geeft dan gewoon zijn input door",
            "no-weights": "Een residual connection heeft geen weights die fout kunnen zijn",
        },
        correct="right",
        why={
            "skipped": "PyTorch draait elk block, elke keer. Het is het block zelf dat kan leren niets te veranderen.",
            "no-weights": "De verbinding zelf heeft geen weights, maar $f$ wel. Waar het om gaat: het block telt alleen $f(x)$ op bij zijn input.",
            "right": "Met $f(x)$ dicht bij 0 ligt $y = x + f(x)$ dicht bij $x$: het block doet bijna niets, en het netwerk doet wat het zonder dat block deed. Zonder de residual connection zou een block met zijn weights moeten leren zijn input precies door te geven, en dat is veel moeilijker.",
        },
    )
    choice_harmless
    return (choice_harmless,)


@app.cell(hide_code=True)
def text_exercise_chain():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Terug door een keten met residual connections

        Schrijf `residual_chain_gradient(ws, x)`: een keten van layers, één voor elke weight in de lijst `ws`, waarin elke layer `h + sigmoid(w * h)` geeft, met `h` wat de layer ervoor gaf, te beginnen met `x`. Geef de gradient terug die bij `x` aankomt, te beginnen met 1 bovenaan, met het recept. `sigmoid` staat al in het vak. De check roept `residual_chain_gradient([2.0, 1.0, 1.5], 0.5)` aan, en dat moet ongeveer `1.747` geven; hij probeert ook andere weights en meer layers.
        """),
        L.task("Laat `residual_chain_gradient` de gradient geven die door de keten met residual connections bij `x` aankomt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_chain():
    box_chain = L.code_box("residual-chain", lines=14)
    box_chain
    return (box_chain,)


@app.cell(hide_code=True)
def check_chain(box_chain):
    result_chain = L.check("residual-chain", box_chain.value)
    result_chain
    return (result_chain,)


@app.cell(hide_code=True)
def hints_chain():
    L.hints(
        "residual-chain",
        [
            "Zoals in het uitgewerkte voorbeeld, maar met de weight van elke layer uit `ws`: bewaar in de forward pass elke `z` in een lijst.",
            "Ga op de terugweg door de layers, de laatste eerst, `for k in reversed(range(len(ws))):`, met de weight `ws[k]` en de `zs[k]` van elke layer, en de factor 1 plus de weight keer de helling.",
        ],
        why="Met de weights 2, 1 en 1,5 en de input 0,5 zijn de $z$'s 1, 1,231 en 3,008. De factoren zijn $1 + 2 \\times 0{,}1966 = 1{,}3932$, $1 + 1 \\times 0{,}1749 = 1{,}1749$ en $1 + 1{,}5 \\times 0{,}04487 = 1{,}0673$, en hun product is ongeveer 1,747.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_block():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een block met een residual connection

        Schrijf een class `ResidualBlock(nn.Module)`: `ResidualBlock(d, hidden)` heeft twee layers, `self.lin1 = nn.Linear(d, hidden)` en `self.lin2 = nn.Linear(hidden, d)`, in die volgorde gemaakt, zodat ze dezelfde willekeurige weights krijgen als in de oplossing van de course. Zijn `forward(x)` geeft `x + f(x)`, met `f` eerst `lin1`, dan ReLU, dan `lin2`: het block uit de figuur. De check maakt `ResidualBlock(4, 8)` na `torch.manual_seed(0)` en vergelijkt wat het geeft voor 3 willekeurige voorbeelden; daarna probeert hij andere groottes, een block dat twee keer na elkaar wordt toegepast, en of `x` zelf blijft zoals hij was.
        """),
        L.task("Laat `ResidualBlock` zijn input plus f van zijn input geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_block():
    box_block = L.code_box("residual-block", lines=12)
    box_block
    return (box_block,)


@app.cell(hide_code=True)
def check_block(box_block):
    result_block = L.check("residual-block", box_block.value)
    result_block
    return (result_block,)


@app.cell(hide_code=True)
def hints_block():
    L.hints(
        "residual-block",
        [
            "`lin2` gaat van `hidden` terug naar `d`, zodat `f(x)` evenveel getallen heeft als `x`, zoals stap 1 van het recept vraagt.",
            "Lees `f` in `forward` van binnen naar buiten: eerst `self.lin1`, dan `torch.relu`, dan `self.lin2`. Tel daarna `x` erbij op, met niets erna.",
        ],
        why="`__init__`: `super().__init__()`, `self.lin1 = nn.Linear(d, hidden)` en `self.lin2 = nn.Linear(hidden, d)`. `forward`: `return x + self.lin2(torch.relu(self.lin1(x)))`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het product van factoren uit [Vanishing en exploding gradients](/course/open/nn-vanishing-exploding?lang=nl) is er nog steeds, maar elke factor is nu $1 + f'$: de somregel uit [Differentieerregels](/course/open/math-derivative-rules?lang=nl) geeft elk block een directe weg terug. Je `ResidualBlock` heeft de vorm van een transformer block: daar is $f$ attention of een MLP, en loopt de input langs allebei. In de volgende les, *Layer normalization*, standaardiseer je elk voorbeeld over zijn eigen features, en zie je waar dat in zo'n block komt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_harmless, result_chain, result_block):
    L.wrap_up(choice_harmless, result_chain, result_block)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
