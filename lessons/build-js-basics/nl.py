# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-js-basics. Route: course/route/lessen/bouwen.yaml. Checks, het plaatje en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-js-basics", "nl")
    C = L.checks   # de twee vakken met JavaScript, die je browser draait, het plaatje van await, en de demo


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Een vraag over [AI op je eigen apparaat](/course/open/build-on-device?lang=nl), en een over [Functies](/course/open/py-functions?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_device():
    quick_device = L.choice(
        "quick-device",
        question="Waarom blijft een model in de browser werken zonder netwerk?",
        options={
            "device": "Het draait op het apparaat, na één download",
            "copy": "De browser bewaart een kopie van het hele internet",
            "phone": "De telefoonmaatschappij houdt de verbinding open",
        },
        correct="device",
        why={
            "device": "Het model kwam één keer over; daarna wordt elke vraag op het apparaat beantwoord, en hoeft er niets over het netwerk.",
            "copy": "Daar heeft geen browser ruimte voor. Waar rekent het model zijn antwoord uit? Zie [AI op je eigen apparaat](/course/open/build-on-device?lang=nl).",
            "phone": "Zonder bereik is er geen verbinding om open te houden. Waar rekent het model? Zie [AI op je eigen apparaat](/course/open/build-on-device?lang=nl).",
        },
    )
    quick_device
    return (quick_device,)


@app.cell(hide_code=True)
def quick_check_return():
    quick_return = L.choice(
        "quick-return",
        question="Wat doet `return` in een functie in Python?",
        options={
            "show": "Het laat een waarde zien op het scherm",
            "again": "Het begint de functie opnieuw vanaf de eerste regel",
            "back": "Het geeft een waarde terug aan wie hem aanriep",
        },
        correct="back",
        why={
            "back": "`return` geeft de waarde aan de code die de functie aanriep, en stopt de functie daar.",
            "show": "Dat doet `print`. Wat krijgt de code die de functie aanriep? Zie [Functies](/course/open/py-functions?lang=nl).",
            "again": "Een functie begint alleen bovenaan als hij opnieuw wordt aangeroepen. Wat geeft `return` door? Zie [Functies](/course/open/py-functions?lang=nl).",
        },
    )
    quick_return
    return (quick_return,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een browser draait één taal: JavaScript. Elke webpagina met een model erin roept dat model aan in JavaScript, of het model nu uit een library komt of in de browser is ingebouwd. Je hoeft er geen webdeveloper voor te worden. Drie dingen zijn genoeg om die code te lezen en aan te passen: hoe een bestand een library laadt, hoe `await` op een antwoord wacht terwijl de pagina blijft reageren, en hoe een web worker naast de pagina rekent.

    **Een voorbeeld uit het dagelijks leven.** In een restaurant brengt de ober een bestelling naar de keuken, en blijft niet bij de keukendeur staan tot het gerecht klaar is: de ober helpt andere tafels, en komt terug als de keuken roept. De pagina is de ober, een lange klus is de keuken. Dat is maar een vergelijking; hieronder zie je hoe JavaScript het doet.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **JavaScript** | de programmeertaal die elke webbrowser draait; met Node.js draait hij ook buiten de browser. Met Java heeft hij niets te maken |
    | **module** | een bestand met JavaScript dat met `import` laadt wat het nodig heeft uit andere modules, en met `export` zijn eigen functies aanbiedt |
    | **promise** | een waarde die later komt, zoals het antwoord op een request: JavaScript geeft je de promise meteen, en de waarde als die er is |
    | **await** | wacht in een functie met `async` ervoor tot een promise zijn waarde geeft; terwijl hij wacht, blijft de pagina reageren op klikken en typen |
    | **web worker** | een stuk JavaScript dat een pagina naast zichzelf draait; zwaar werk daarin, zoals een model dat rekent, laat de pagina niet vastlopen. Pagina en worker praten door elkaar berichten te sturen |

    ### Python en JavaScript naast elkaar

    | Python | JavaScript |
    |---|---|
    | `x = 3` | `let x = 3;`, of `const x = 3;` voor een naam die zijn waarde houdt |
    | `def twice(x):` met `return 2 * x` ingesprongen eronder | `function twice(x) { return 2 * x; }` |
    | `[1, 2, 3]` en `len(xs)` | `[1, 2, 3]` en `xs.length` |
    | `{"model": "qwen3-0.6b"}` en `d["model"]` | `{model: "qwen3-0.6b"}` en `d.model` |
    | `for x in xs:` | `for (const x of xs) { ... }` |
    | `for i in range(len(xs)):` | `for (let i = 0; i < xs.length; i++) { ... }` |
    | `if x > 0:` en `else:` | `if (x > 0) { ... } else { ... }` |
    | `total += x` | `total += x;` |
    | `print(x)` | `console.log(x);` |
    | `None` | `null`, en `undefined` voor een waarde die nooit gegeven is |
    | `json.loads(text)` | `JSON.parse(text)` |

    Accolades `{ }` omsluiten een blok waar Python inspringt, en een opdracht eindigt op `;`. `i++` telt 1 op bij `i`.

    ### Wachten zonder vast te lopen
    """)
    return


@app.cell(hide_code=True)
def picture_await():
    mo.Html(C.await_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_await():
    mo.vstack([
        mo.md(r"""
        `fetch(url)` stuurt een request, zoals `urllib.request.urlopen` in Python, maar geeft meteen een promise terug. `await` wacht op de waarde ervan; intussen is de pagina vrij. `urlopen` in Python wacht ook, maar dan draait er verder niets. En een module laadt een library via het adres ervan:

        ```javascript
        import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers";
        ```

        Die regel laadt `pipeline` uit Transformers.js, een library die de volgende les gebruikt (het adres zoals in november 2026); een echte pagina zet een vast versienummer achter de naam, zodat een nieuwe versie van de library de pagina niet verandert.

        `await` maakt de pagina vrij terwijl je code wacht, maar niet terwijl je code rekent. Een model dat twee seconden op de pagina rekent, legt de pagina twee seconden stil. Dat werk gaat naar een web worker: de pagina stuurt hem een bericht met de invoer, en de worker stuurt een bericht terug met het antwoord.
        """),
        L.watch_out("JavaScript heeft ook een loop `for ... in`, en die gaat over de posities van een array als tekst, `\"0\"`, `\"1\"`, niet over de waarden. Voor de waarden gebruik je `for ... of`."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: AI-code in JavaScript lezen

    1. **De imports bovenaan:** welke library laadt de pagina, en waarvandaan?
    2. **Elke `await`:** elk is een wachtmoment, op een download, een model dat laadt, of een antwoord.
    3. **`new Worker` en `postMessage`:** werk dat naast de pagina draait, en de berichten erheen en terug.
    4. **De rest:** functies, arrays, objecten en loops, zoals in Python, met accolades.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een functie die een lijst teksten als JSON ophaalt, en teruggeeft hoeveel tekens ze gemiddeld hebben:

    ```javascript
    async function averageLength(url) {
      const response = await fetch(url);
      const texts = await response.json();
      let total = 0;
      for (const text of texts) {
        total += text.length;
      }
      return total / texts.length;
    }
    ```

    1. **`async function`:** een functie die mag wachten; hij geeft zelf een promise terug, dus wie hem aanroept, gebruikt ook `await`.
    2. **`await fetch(url)`:** het request; `response` is het antwoord, zodra het er is.
    3. **`await response.json()`:** de body als JSON lezen kost ook even, dus ook dat geeft een promise, en `texts` krijgt de lijst.
    4. **De loop:** `for (const text of texts)` gaat over de teksten, en `text.length` is het aantal tekens van elk.
    5. **Het gemiddelde:** het totaal gedeeld door het aantal teksten. Voor `["red", "blue"]` is dat (3 + 4) / 2: `3.5`.
    """)
    return


@app.cell(hide_code=True)
def text_dot():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Paar voor paar

        Schrijf `dot(a, b)` in JavaScript: vermenigvuldig de getallen van twee even lange arrays paar voor paar, en tel alles op. `dot([1, 2, 3], [4, 5, 6])` geeft `32`: 1 × 4 + 2 × 5 + 3 × 6. Netwerken die cijfers of tekst lezen, rekenen deze som vele keren uit; de volgende les gebruikt hem om een klein netwerk in de browser te laten draaien. Je browser draait je functie, in een web worker, met de gevallen van de check.
        """),
        L.task("Maak `dot` af: loop door de posities, en tel de producten op."),
    ])
    return


@app.cell(hide_code=True)
def exercise_dot():
    box_dot = L.code_box("dot", lines=8)
    box_dot
    return (box_dot,)


@app.cell(hide_code=True)
def run_dot_cell(box_dot):
    run_dot = L.js_run("dot", box_dot.value)
    run_dot
    return (run_dot,)


@app.cell(hide_code=True)
def check_dot(run_dot):
    result_dot = L.check("dot", run_dot)
    result_dot
    return (result_dot,)


@app.cell(hide_code=True)
def hints_dot():
    L.hints(
        "dot",
        [
            "Begin met een totaal van 0, en loop door de posities van `a` met een `for`-loop die telt, zoals in de tabel hierboven.",
            "`for (let i = 0; i < a.length; i++) { ... }` gaat langs elke positie; tel tussen de accolades `a[i] * b[i]` op bij het totaal.",
            "`let total = 0;` vóór de loop, `total += a[i] * b[i];` erin, en `return total;` erna.",
        ],
        why="""```javascript
function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}
```

`let` maakt het totaal, de loop telt `i` van 0 tot de laatste positie, en `+=` telt elk product erbij op.""",
    )
    return


@app.cell(hide_code=True)
def text_sum():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Wacht erop

        Schrijf `async function sumFrom(url)`: haal de JSON op van `url`, een lijst getallen, en geef hun som terug. De check geeft je functie adressen die beginnen met `data:`, zoals `"data:application/json,[1,2,3]"`: de JSON staat in het adres zelf, dus `fetch` heeft geen netwerk nodig. Voor dat adres geeft `sumFrom` als uitkomst `6`.
        """),
        L.task("Maak `sumFrom` af: ophalen, dan de JSON, elk met `await`, en dan de lijst optellen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_sum():
    box_sum = L.code_box("sumFrom", lines=10)
    box_sum
    return (box_sum,)


@app.cell(hide_code=True)
def run_sum_cell(box_sum):
    run_sum = L.js_run("sumFrom", box_sum.value)
    run_sum
    return (run_sum,)


@app.cell(hide_code=True)
def check_sum(run_sum):
    result_sum = L.check("sumFrom", run_sum)
    result_sum
    return (result_sum,)


@app.cell(hide_code=True)
def hints_sum():
    L.hints(
        "sumFrom",
        [
            "Volg het uitgewerkte voorbeeld: eerst het antwoord van `fetch`, dan de JSON ervan, dan een loop over de lijst.",
            "`const numbers = await response.json();` geeft de lijst. Tel daarna de getallen op met `for (const number of numbers)`.",
            "`const response = await fetch(url);` en `const numbers = await response.json();`, dan `let total = 0;`, de loop met `total += number;`, en `return total;`.",
        ],
        why="""```javascript
async function sumFrom(url) {
  const response = await fetch(url);
  const numbers = await response.json();
  let total = 0;
  for (const number of numbers) {
    total += number;
  }
  return total;
}
```

Zowel `fetch` als `.json()` geeft een promise; `await` wacht op elke waarde voordat de volgende regel hem gebruikt.""",
    )
    return


@app.cell(hide_code=True)
def text_worker():
    mo.md(r"""
    ### Oefening 3. Naast de pagina
    """)
    return


@app.cell(hide_code=True)
def choice_worker_cell():
    choice_worker = L.choice(
        "worker",
        question="Een pagina draait een model dat voor elk antwoord twee seconden rekent. Waarom draait de pagina het model in een web worker?",
        options={
            "faster": "Een worker rekent sneller dan de pagina zelf",
            "free": "Zodat de pagina op klikken blijft reageren",
            "download": "Alleen een worker mag een model downloaden",
        },
        correct="free",
        why={
            "free": "De worker rekent naast de pagina, dus de pagina blijft vrij voor de gebruiker; de pagina krijgt het antwoord als bericht.",
            "faster": "Het werk is hetzelfde, op dezelfde computer. Wat verandert er voor de pagina terwijl de worker rekent?",
            "download": "Een pagina kan zelf een model downloaden, met `fetch`. Wat zouden twee seconden rekenen met de pagina doen?",
        },
    )
    choice_worker
    return (choice_worker,)


@app.cell(hide_code=True)
def text_import():
    mo.md(r"""
    ### Oefening 4. Een library laden
    """)
    return


@app.cell(hide_code=True)
def choice_import_cell():
    choice_import = L.choice(
        "import",
        question="Een webpagina heeft de functie `pipeline` nodig uit de module op `https://cdn.jsdelivr.net/npm/@huggingface/transformers`. Welke regel laadt hem?",
        options={
            "module": "`import { pipeline } from \"https://cdn.jsdelivr.net/npm/@huggingface/transformers\";`",
            "python": "`from \"https://cdn.jsdelivr.net/npm/@huggingface/transformers\" import { pipeline };`",
            "require": "`const { pipeline } = require(\"https://cdn.jsdelivr.net/npm/@huggingface/transformers\");`",
        },
        correct="module",
        why={
            "module": "Een module importeert een naam, tussen accolades, uit een andere module, via zijn adres.",
            "python": "Dat is de volgorde van `from ... import` in Python. Wat komt er in JavaScript eerst?",
            "require": "Met `require` laadde Node.js code voordat JavaScript modules had, en een browser kent het niet. Hoe laadt een module een naam?",
        },
    )
    choice_import
    return (choice_import,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### De pagina, of een web worker

    Twee keer hetzelfde werk: één keer op de pagina zelf, één keer in een web worker. Na een klik beweegt de stip een seconde, dan begint het werk, en de stip beweegt door zolang de pagina hem kan tekenen; de demo meet de langste pauze tussen twee tekeningen, op je eigen computer.
    """)
    return


@app.cell(hide_code=True)
def demo_freeze():
    _words = {"page": "Reken op de pagina", "worker": "Reken in een web worker", "busy": "Bezig met rekenen …",
              "result": "{where}: het werk duurde {work} ms, en de stip stond hoogstens {gap} ms stil.",
              "where_page": "Op de pagina", "where_worker": "In een web worker", "frames": "{n} tekeningen tot nu toe"}
    mo.ui.anywidget(C.freeze(_words, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Een functie in JavaScript werkt zoals de functies uit [Functies](/course/open/py-functions?lang=nl): hij krijgt waarden en geeft er een terug met `return`. `fetch` en `await` doen hetzelfde soort request als de aanroepen van een model in deze module, en een web worker houdt de pagina vrij terwijl een model op het apparaat rekent, zoals in [AI op je eigen apparaat](/course/open/build-on-device?lang=nl). In de volgende les, *Je model in de browser*, gaat een klein netwerk dat cijfers leest als JSON naar de browser, en maakt een functie zoals je `dot` daar zijn sommen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_import, choice_worker, result_dot, result_sum):
    L.wrap_up(result_dot, result_sum, choice_worker, choice_import)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
