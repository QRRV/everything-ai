# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-model-in-browser. Route: course/route/lessen/bouwen.yaml. Checks, het plaatje en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-model-in-browser", "nl")
    C = L.checks   # het cijfernetwerk, de check van as_json, de twee vakken met JavaScript, het plaatje en de demo


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Een vraag over [Net genoeg JavaScript](/course/open/build-js-basics?lang=nl), een over [Project: een MLP die handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl) en een over [Data als tekst: JSON](/course/open/py-json?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_loop():
    quick_loop = L.choice(
        "quick-loop",
        question="Langs welke posities van de array `a` gaat `for (let i = 0; i < a.length; i++)`?",
        options={
            "one": "Van 1 tot en met `a.length`",
            "all": "Langs elke positie, van 0 tot de laatste",
            "values": "Langs geen enkele: hij gaat langs de waarden",
        },
        correct="all",
        why={
            "all": "`i` begint bij 0 en gaat door zolang `i < a.length`, dus hij stopt na de laatste positie, `a.length - 1`.",
            "one": "Waar begint `i`, en wat doet `i < a.length` aan het eind? Zie [Net genoeg JavaScript](/course/open/build-js-basics?lang=nl).",
            "values": "Dat doet `for (const x of a)`. Wat is `i` hier? Zie [Net genoeg JavaScript](/course/open/build-js-basics?lang=nl).",
        },
    )
    quick_loop
    return (quick_loop,)


@app.cell(hide_code=True)
def quick_check_mlp():
    quick_mlp = L.choice(
        "quick-mlp",
        question="Wat geeft je `DigitsMLP` voor één afbeelding van een cijfer?",
        options={
            "digit": "Het cijfer zelf, één getal van 0 tot en met 9",
            "pixels": "784 getallen, één voor elke pixel",
            "logits": "Tien logits, één voor elk cijfer",
        },
        correct="logits",
        why={
            "logits": "Zijn output layer heeft tien neuronen; het cijfer dat hij leest, is dat met de grootste logit.",
            "digit": "Wat komt er uit de output layer, en hoeveel? Zie [Project: een MLP die handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl).",
            "pixels": "Die gaan erin. Wat komt er uit de output layer? Zie [Project: een MLP die handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl).",
        },
    )
    quick_mlp
    return (quick_mlp,)


@app.cell(hide_code=True)
def quick_check_json():
    quick_json = L.choice(
        "quick-json",
        question='Wat geeft `json.dumps({"w": [1, 2]})`?',
        options={
            "text": 'De tekst `{"w": [1, 2]}`',
            "dict": 'Weer de dict `{"w": [1, 2]}`',
            "list": "De lijst `[1, 2]`",
        },
        correct="text",
        why={
            "text": "`json.dumps` schrijft een dict als tekst, die een programma in een andere taal kan lezen, zoals een pagina in JavaScript.",
            "dict": "Wat voor waarde maakt `json.dumps`? Zie [Data als tekst: JSON](/course/open/py-json?lang=nl).",
            "list": "`json.dumps` houdt de hele dict. Wat voor waarde maakt het ervan? Zie [Data als tekst: JSON](/course/open/py-json?lang=nl).",
        },
    )
    quick_json
    return (quick_json,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Het netwerk dat cijfers leest, uit de projectles, bestaat uit 50.890 getallen. Een webpagina kan die getallen meekrijgen, en de sommen van het netwerk zelf maken, in JavaScript: dan leest de pagina een cijfer zonder server, zonder API key, en de tekening verlaat de computer nooit. Het is de on-device AI van [AI op je eigen apparaat](/course/open/build-on-device?lang=nl), met een netwerk dat je door en door kent. Libraries doen hetzelfde met grote open modellen, en Chrome heeft er zelfs een ingebouwd.

    **Een voorbeeld uit het dagelijks leven.** Een recept is hetzelfde op papier, op een telefoon of op de koelkastdeur: wie het heeft, kan koken zonder de chef te vragen. De weights zijn het recept van een netwerk. Dat is maar een vergelijking; hieronder zie je hoe het werkt.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **Transformers.js** | een JavaScript-library van Hugging Face die getrainde modellen uit zijn model hub in de browser laat draaien, met pipelines zoals in `transformers` voor Python |
    | **WebLLM** | een library die large language models helemaal in de browser laat draaien, op de GPU van de computer, met een interface zoals de OpenAI API |
    | **ingebouwde browser-AI** | een model dat de browser zelf meebrengt, en dat elke pagina kan gebruiken zonder zelf een model te downloaden, zoals Gemini Nano in Chrome via zijn Prompt API |
    | **WebGPU** | de manier waarop een pagina de GPU van de computer laat rekenen, waardoor een model in de browser vele malen sneller draait dan op de CPU |
    | **ONNX** | een open bestandsformaat voor getrainde modellen, zodat een model uit PyTorch in andere software kan draaien, zoals een browser |
    | `.tolist()` | maakt van een tensor een lijst Python-getallen, een lijst van lijsten bij een tensor met twee assen, en die kan `json.dumps` schrijven |

    ### Van PyTorch naar de pagina
    """)
    return


@app.cell(hide_code=True)
def picture_way():
    mo.Html(C.way_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_layers():
    mo.vstack([
        mo.md(r"""
        Een layer `nn.Linear` heeft een tabel met weights en een lijst biases. Rij `j` van `model.hidden.weight` bevat de 784 weights van hidden neuron `j`, dus de tabel heeft 64 rijen van 784 getallen; `model.hidden.bias` heeft 64 getallen. Voor een input `x` geeft neuron `j` als uitkomst `dot(weight[j], x) + bias[j]`: je `dot` uit de vorige les, plus een bias. `forward` van `DigitsMLP` doet dat voor de hidden layer, haalt de sommen door ReLU, en doet het nog eens voor de output layer, die tien logits geeft.

        De pagina heeft dus vier dingen nodig: `w1` en `b1` van de hidden layer, `w2` en `b2` van de output layer. `json.dumps` kan geen tensor schrijven, maar `.tolist()` maakt er eerst een lijst van: `json.dumps({"b": torch.tensor([0.5, -1.0]).tolist()})` geeft de tekst `{"b": [0.5, -1.0]}`. Als JSON-tekst leest een pagina ze met `await (await fetch(url)).json()`. In JavaScript is ReLU `Math.max(0, v)`, zoals `max(0, v)` in Python, en `out.push(v)` zet `v` achteraan een array, zoals `append` in Python.

        Als tekst is het netwerk ongeveer 1,1 MB, vijfenhalf keer zo groot als in `float32` (0,2 MB): JSON schrijft elk getal met al zijn cijfers. Bij zo'n klein netwerk maakt dat niet uit. Een groot model gaat in plaats daarvan in een binair formaat naar de browser, zoals ONNX.
        """),
        L.watch_out("Een pagina die zijn weights met `fetch` laadt, downloadt ze voor elke gebruiker, en een browser bewaart ze maar zo lang als zijn cache. Transformers.js en WebLLM bewaren wat ze downloadden in de eigen opslag van de browser, zodat een tweede bezoek ze niet opnieuw downloadt."),
    ])
    return


@app.cell(hide_code=True)
def text_large():
    mo.md(r"""
    ### Grote modellen in de browser

    Hetzelfde idee, weights in de pagina en de sommen in JavaScript, werkt voor grote modellen, maar niemand schrijft die sommen met de hand. Drie manieren (voorbeelden in november 2026):

    | Manier | Wat er draait | Code |
    |---|---|---|
    | Transformers.js | een open model van Hugging Face, omgezet naar ONNX, voor een taak: embeddings, spraak naar tekst, afbeeldingen | `const extractor = await pipeline("feature-extraction", "mixedbread-ai/mxbai-embed-xsmall-v1", { device: "webgpu" });` |
    | WebLLM | een open chat model met 4 bits, zoals Qwen3-0.6B, op de GPU via WebGPU | `const engine = await CreateMLCEngine("Qwen3-0.6B-q4f16_1-MLC");` en dan `await engine.chat.completions.create({ messages })` |
    | de Prompt API van Chrome | Gemini Nano, ingebouwd in Chrome: de pagina brengt geen model mee | `const session = await LanguageModel.create();` en dan `await session.prompt("...")` |

    Met Transformers.js en WebLLM kiest de pagina het model, en downloadt elke gebruiker het één keer. Met de Prompt API downloadt Chrome Gemini Nano één keer voor alle pagina's; de pagina kan het model niet kiezen, en het werkt alleen waar Chrome het aanbiedt: in Chrome 148 en later, op een computer met Windows, macOS of Linux of op een Chromebook Plus, met minstens 22 GB vrij op de schijf en een GPU met minstens 4 GB eigen geheugen, of 16 GB geheugen en 4 CPU-kernen. Een pagina vraagt het eerst na met `await LanguageModel.availability()`, en heeft een plan voor als het antwoord `"unavailable"` is.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: je eigen netwerk in de browser

    1. **Weights als JSON:** elke tensor `.tolist()`, samen in een dict, `json.dumps`.
    2. **In de pagina:** laad de JSON-tekst met `fetch`, en lees hem met `.json()`.
    3. **Elke layer:** voor elk neuron `dot` van zijn rij weights met de input, plus zijn bias.
    4. **Tussen de layers:** ReLU, `Math.max(0, v)`, net als in `forward` van je netwerk.
    5. **Het antwoord:** de positie van de grootste logit. Voor een groot model gebruik je in plaats daarvan Transformers.js, WebLLM of de Prompt API.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een netwerk met twee inputs, twee hidden neuronen en twee outputs, zoals de pagina het leest:

    ```javascript
    const net = {w1: [[1, 1], [-1, 2]], b1: [-1, 0], w2: [[2, -1], [0, 1]], b2: [0, 0.5]};
    ```

    Wat maakt het van de input `[3, 0]`?

    1. **Hidden neuron 0:** zijn rij is `[1, 1]`, zijn bias `-1`: 1 × 3 + 1 × 0 − 1 = `2`.
    2. **Hidden neuron 1:** zijn rij is `[-1, 2]`, zijn bias `0`: −1 × 3 + 2 × 0 + 0 = `-3`.
    3. **ReLU:** `2` blijft `2`, en `-3` wordt `0`; de hidden layer geeft `[2, 0]`.
    4. **De output layer:** 2 × 2 − 1 × 0 + 0 = `4`, en 0 × 2 + 1 × 0 + 0,5 = `0.5`: de logits zijn `[4, 0.5]`.
    5. **Het antwoord:** de grootste logit staat op positie 0, dus het netwerk zegt 0.
    """)
    return


@app.cell(hide_code=True)
def text_json():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De weights als tekst

        Schrijf `as_json(model)`: het geeft de weights en biases van een `DigitsMLP` als JSON-tekst, met de keys `"w1"` en `"b1"` voor `model.hidden`, en `"w2"` en `"b2"` voor `model.out`. Voor `DigitsMLP(3)` bevat `"w1"` 3 lijsten van 784 getallen. De check roept je functie aan met netwerken van een paar groottes, en vergelijkt de getallen met de tensors.
        """),
        L.task("Maak `as_json` af: zet de vier tensors als lijsten in een dict, en geef die terug als JSON-tekst."),
    ])
    return


@app.cell(hide_code=True)
def exercise_json():
    box_json = L.code_box("as_json", lines=12)
    box_json
    return (box_json,)


@app.cell(hide_code=True)
def check_json(box_json):
    result_json = L.check("as_json", box_json.value)
    result_json
    return (result_json,)


@app.cell(hide_code=True)
def hints_json():
    L.hints(
        "as_json",
        [
            "`json.dumps` kan dicts, lijsten en getallen schrijven, maar geen tensors: maak eerst van elke tensor een lijst.",
            "`model.hidden.weight.tolist()` geeft de weights van de hidden layer als lijst van lijsten. Zet de vier lijsten in een dict met de keys `\"w1\"`, `\"b1\"`, `\"w2\"` en `\"b2\"`.",
            "`weights = {\"w1\": model.hidden.weight.tolist(), \"b1\": model.hidden.bias.tolist(), ...}` met `model.out` voor `\"w2\"` en `\"b2\"`, dan `return json.dumps(weights)`.",
        ],
        why="""```python
import json


def as_json(model):
    weights = {
        "w1": model.hidden.weight.tolist(),
        "b1": model.hidden.bias.tolist(),
        "w2": model.out.weight.tolist(),
        "b2": model.out.bias.tolist(),
    }
    return json.dumps(weights)
```

`.tolist()` maakt van elke tensor een lijst getallen, de dict geeft elk zijn naam, en `json.dumps` schrijft het allemaal als tekst.""",
    )
    return


@app.cell(hide_code=True)
def text_layer():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Eén layer in JavaScript

        Schrijf `layer(w, b, x)` in JavaScript: voor elke rij `w[i]` geeft het `dot(w[i], x) + b[i]`, allemaal in één array. Het vak begint met een `dot` zoals die van jou uit de vorige les. `layer([[1, 2], [3, -1]], [0.5, -2], [2, 1])` geeft `[4.5, 3]`: 1 × 2 + 2 × 1 + 0,5, en 3 × 2 − 1 × 1 − 2. Je browser draait je functie, in een web worker.
        """),
        L.task("Maak `layer` af: één getal per rij van `w`, elk met `push` achteraan een array."),
    ])
    return


@app.cell(hide_code=True)
def exercise_layer():
    box_layer = L.code_box("layer", lines=16)
    box_layer
    return (box_layer,)


@app.cell(hide_code=True)
def run_layer_cell(box_layer):
    run_layer = L.js_run("layer", box_layer.value)
    run_layer
    return (run_layer,)


@app.cell(hide_code=True)
def check_layer(run_layer):
    result_layer = L.check("layer", run_layer)
    result_layer
    return (result_layer,)


@app.cell(hide_code=True)
def hints_layer():
    L.hints(
        "layer",
        [
            "Begin met een lege array, en loop langs de rijen van `w` met een `for`-loop die telt, zoals in `dot`.",
            "`for (let i = 0; i < w.length; i++) { ... }` gaat langs de neuronen; zet daarbinnen `dot(w[i], x) + b[i]` met `push` achteraan de array.",
            "`const out = [];` vóór de loop, `out.push(dot(w[i], x) + b[i]);` erin, en `return out;` erna.",
        ],
        why="""```javascript
function layer(w, b, x) {
  const out = [];
  for (let i = 0; i < w.length; i++) {
    out.push(dot(w[i], x) + b[i]);
  }
  return out;
}
```

Elke rij van `w` is één neuron: `dot` met de input, plus zijn bias, geeft wat dat neuron geeft.""",
    )
    return


@app.cell(hide_code=True)
def text_forward():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Het hele netwerk

        Schrijf `forward(net, pixels)` in JavaScript, zoals `forward` van je `DigitsMLP`: de hidden layer met `net.w1` en `net.b1`, ReLU op wat die geeft, dan de output layer met `net.w2` en `net.b2`, waarvan de getallen terugkomen zoals ze zijn. De check zet de `dot` en `layer` van de course vóór je code, dus je kunt ze hier aanroepen; in de demo hieronder doet je eigen `layer` uit oefening 2 het werk. Met

        ```javascript
        const net = {w1: [[1, -1], [0.5, 2]], b1: [0, -3], w2: [[1, 1], [2, -1], [-1, 0]], b2: [0, 0.5, -1]};
        ```

        geeft `forward(net, [1, 2])` als uitkomst `[1.5, -1, -1]`.
        """),
        L.task("Maak `forward` af: de hidden layer, ReLU, de output layer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_forward():
    box_forward = L.code_box("forward", lines=10)
    box_forward
    return (box_forward,)


@app.cell(hide_code=True)
def run_forward_cell(box_forward):
    run_forward = L.js_run("forward", box_forward.value)
    run_forward
    return (run_forward,)


@app.cell(hide_code=True)
def check_forward(run_forward):
    result_forward = L.check("forward", run_forward)
    result_forward
    return (result_forward,)


@app.cell(hide_code=True)
def hints_forward():
    L.hints(
        "forward",
        [
            "Volg het uitgewerkte voorbeeld: eerst wat de hidden layer geeft, dan ReLU op elk van die getallen, dan de output layer op het resultaat.",
            "`layer(net.w1, net.b1, pixels)` geeft de sommen van de hidden layer. Loop erlangs met `for (const v of ...)`, en zet `Math.max(0, v)` met `push` achteraan een nieuwe array.",
            "`const hidden = [];`, de loop met `hidden.push(Math.max(0, v));`, en dan `return layer(net.w2, net.b2, hidden);`.",
        ],
        why="""```javascript
function forward(net, pixels) {
  const hidden = [];
  for (const v of layer(net.w1, net.b1, pixels)) {
    hidden.push(Math.max(0, v));
  }
  return layer(net.w2, net.b2, hidden);
}
```

Dezelfde drie stappen als `self.out(torch.relu(self.hidden(x)))` in je `DigitsMLP`: de hidden layer, ReLU, de output layer.""",
    )
    return


@app.cell(hide_code=True)
def text_builtin():
    mo.md(r"""
    ### Oefening 4. Ingebouwd in de browser
    """)
    return


@app.cell(hide_code=True)
def choice_builtin_cell():
    choice_builtin = L.choice(
        "builtin",
        question="Een pagina beantwoordt vragen met de Prompt API van Chrome. Waar komt het model vandaan dat antwoordt?",
        options={
            "hub": "De pagina downloadt het van Hugging Face",
            "google": "Google draait het op zijn servers",
            "chrome": "Chrome brengt het mee, voor elke pagina",
        },
        correct="chrome",
        why={
            "chrome": "Gemini Nano is ingebouwd in Chrome: Chrome downloadt het één keer, en draait het op de computer zelf, voor elke pagina die erom vraagt.",
            "hub": "Dat doet een pagina met Transformers.js of WebLLM. Wat brengt de pagina mee bij de Prompt API?",
            "google": "Dan zou elke vraag over het netwerk gaan. Waar draait een ingebouwd model?",
        },
    )
    choice_builtin
    return (choice_builtin,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Teken een cijfer

    Teken een cijfer in het vierkant. Na elke streek wordt de tekening 28 × 28 pixels, zoals de cijfers van QMNIST gemaakt zijn, en je eigen code leest hem, in je browser: de weights zoals je `as_json` ze schreef, je `layer` en je `forward`. Het netwerk is dat uit de projectles, op dezelfde manier getraind door de course: 64 hidden neuronen, 10 epochs, een learning rate van 0,5. Het leest 929 van de 1.000 testcijfers goed. Onder het vierkant leest PyTorch dezelfde pixels, hier in Python.
    """)
    return


@app.cell(hide_code=True)
def demo_draw(box_forward, box_json, box_layer, result_forward, result_json, result_layer):
    mo.stop(not (result_json.state == result_layer.state == result_forward.state == "correct"),
            mo.md("*Maak eerst oefening 1, 2 en 3: dan gebruikt de demo je `as_json`, je `layer` en je `forward`.*"))
    _net = C.learner_json(box_json.value)
    _words = {"clear": "Opnieuw", "draw": "Teken een cijfer in het vierkant.", "reads": "Je code in je browser leest een {digit}.",
              "stopped": "Je code was na 3 seconden niet klaar, dus je browser stopte hem.", "error": "Je code stopte: {error}"}
    pad = mo.ui.anywidget(C.draw(box_layer.value + "\n" + box_forward.value, _net, _words, L.lang))
    pad
    return (pad,)


@app.cell(hide_code=True)
def demo_torch(pad):
    _v = pad.widget.value
    mo.stop(not _v.get("pixels"), None)
    _digit, _logits = C.torch_reads(_v["pixels"])
    _gap = max(abs(a - b) for a, b in zip(_logits, _v["scores"]))
    mo.md("PyTorch, hier in Python, leest een **{digit}** in dezelfde pixels. Zijn tien logits verschillen minder dan {gap} van die van je code in je browser: hetzelfde netwerk, in twee talen.".format(digit=_digit, gap=L.number(*C.less_than(_gap))))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het netwerk is de `DigitsMLP` van [Project: een MLP die handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl), zijn weights reizen als de JSON van [Data als tekst: JSON](/course/open/py-json?lang=nl), en zijn sommen zijn je `dot` uit [Net genoeg JavaScript](/course/open/build-js-basics?lang=nl), in je browser zoals in [AI op je eigen apparaat](/course/open/build-on-device?lang=nl). Transformers.js en WebLLM doen hetzelfde met modellen die te groot zijn om met de hand te schrijven, en de Prompt API met een model dat in Chrome is ingebouwd. Hiermee is de module *Modellen aanroepen* klaar; daarna komen prompts voor ontwikkelaars.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_builtin, result_forward, result_json, result_layer):
    L.wrap_up(result_json, result_layer, result_forward, choice_builtin)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
