# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-on-device. Route: course/route/lessen/bouwen.yaml. Checks, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-on-device", "nl")
    C = L.checks   # de checks van de twee functies, de plaatjes, en de demo voor deze computer


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Waar een model draait](/course/open/build-where-models-run?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_gb():
    quick_gb = L.choice(
        "quick-gb",
        question="Een model heeft 8 miljard weights van 4 bits. Hoeveel geheugen nemen ze in?",
        options={
            "32": "32 GB",
            "4": "4 GB",
            "8": "8 GB",
        },
        correct="4",
        why={
            "4": "8 miljard keer 4 bits is 32 miljard bits; gedeeld door 8 is dat 4 miljard bytes: 4 GB.",
            "32": "32 miljard is het aantal bits. Hoeveel bits gaan er in een byte? Zie [Waar een model draait](/course/open/build-where-models-run?lang=nl).",
            "8": "Dat zou 8 bits per weight zijn, elk één byte. Hoeveel bytes zijn 4 bits? Zie [Waar een model draait](/course/open/build-where-models-run?lang=nl).",
        },
    )
    quick_gb
    return (quick_gb,)


@app.cell(hide_code=True)
def quick_check_move():
    quick_move = L.choice(
        "quick-move",
        question="Je verhuist je app van Ollama op je computer naar een inference provider. Wat verander je in je aanroep?",
        options={
            "all": "Het hele request: elke plek heeft zijn eigen formaat",
            "nothing": "Niets: hetzelfde adres werkt overal",
            "url": "De base URL, de naam van het model en een key",
        },
        correct="url",
        why={
            "url": "Het request blijft hetzelfde; het adres en de naam van het model veranderen, en een host wil bij elk request een key.",
            "all": "Allebei spreken ze dezelfde OpenAI-compatible API. Wat verschilt er tussen de plekken? Zie [Waar een model draait](/course/open/build-where-models-run?lang=nl).",
            "nothing": "Het adres van Ollama is op je eigen computer; de provider heeft een eigen adres. Zie [Waar een model draait](/course/open/build-where-models-run?lang=nl).",
        },
    )
    quick_move
    return (quick_move,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Tot nu toe draaide elk model op een computer die je requests krijgt: de server van een bedrijf, of de server van de course op je eigen computer. Een model kan ook in de app zelf draaien: in de browser van wie je pagina opent, of op diens telefoon. Dan werkt het in de trein zonder bereik, kost geen enkele vraag je geld, en verlaat een foto of een brief het apparaat nooit. De prijs: het model moet klein genoeg zijn voor dat apparaat, en elke gebruiker downloadt het eerst.

    **Een voorbeeld uit het dagelijks leven.** Een kaart op je telefoon kan elke keer dat je kijkt van internet komen, of je downloadt hem één keer vóór een reis: dan werkt hij ook in de bergen, maar hij neemt ruimte in op je telefoon, en je downloadt hem opnieuw als de wegen veranderen. Dat is maar een vergelijking; hieronder zie je de echte afweging.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **on-device AI** | een model dat draait op het apparaat van wie het gebruikt, zoals een telefoon, een laptop of de browser daarop, in plaats van op een server; het werkt zonder netwerk, en de data blijft op het apparaat |
    | **apparaat** | hier: de telefoon, laptop of andere computer van wie je app gebruikt; in code van PyTorch, later in de course, betekent `device` waar een tensor wordt berekend, op de CPU of de GPU |
    | **Mbit/s** | megabits per seconde: hoe snel een verbinding is; de grootte van een bestand is in megabytes, MB, en een byte heeft 8 bits |
    | **TensorFlow.js** | een library van Google die modellen in de browser traint en laat draaien, in JavaScript, de taal van webpagina's |
    | **Pyodide** | Python dat in een webbrowser kan draaien, met packages als NumPy en pandas, zodat Python-code in een pagina draait zonder server |

    ### Wat er over het netwerk gaat
    """)
    return


@app.cell(hide_code=True)
def picture_network():
    mo.Html(C.network_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_trade():
    mo.vstack([
        mo.md(r"""
        Met het model op een server gaat elke vraag met zijn data over het netwerk, en komt elk antwoord terug: daar is elke keer een netwerk voor nodig, en wie de server draait, kan de data zien. Met het model op het apparaat komt alleen het model over, één keer; daarna gaat er niets meer weg.

        | | Op een server | Op het apparaat |
        |---|---|---|
        | zonder netwerk | werkt niet | werkt |
        | wat een vraag kost | per token, of de server | niets: het apparaat van de gebruiker doet het werk |
        | wie de data ziet | wie de server draait | alleen de gebruiker |
        | hoe groot het model kan zijn | zo groot als je kunt betalen | klein genoeg voor het apparaat |
        | vóór de eerste vraag | niets | de download |
        | de hardware | jij kiest hem | de gebruiker heeft hem: van een oude telefoon tot een nieuwe laptop |

        On-device AI is al overal (voorbeelden in november 2026): het toetsenbord van een telefoon stelt het volgende woord voor, dicteren maakt van spraak tekst, en vertaalapps vertalen zonder netwerk als je een taal hebt gedownload. Browsers krijgen eigen modellen ingebouwd; een latere les van deze module roept er een aan.

        ### Past het, en hoe lang duurt het?

        De weights nemen params × bits / 8 bytes in, zoals in [Waar een model draait](/course/open/build-where-models-run?lang=nl). Op een apparaat kies je de bits: met minder bits worden de weights kleiner, en een beetje minder precies. Meer bits dan waarmee het model gemaakt is, voegen niets toe: Qwen3-0.6B komt in `bfloat16`, dus meer dan 16 bits heeft het niets aan. En een apparaat heeft ook geheugen nodig voor al het andere dat het doet.

        | Model | Weights | Grootte van de weights |
        |---|---|---|
        | een klein model dat handgeschreven cijfers leest | 50.890 | 0,2 MB in `float32` |
        | Qwen3-0.6B | 0,6 miljard | 1.200 MB in `bfloat16` |
        | een model van 8 miljard | 8 miljard | 4.000 MB met 4 bits |

        Elke gebruiker downloadt het model één keer, en opnieuw bij elke nieuwe versie. Verbindingen worden opgegeven in megabits per seconde, groottes in megabytes, dus in de berekening zit een factor 8.

        ### De tools (voorbeelden in november 2026)

        | Waar het draait | Tool | Wat het doet |
        |---|---|---|
        | in de browser | TensorFlow.js | traint modellen en laat ze draaien in JavaScript, op de GPU van de computer als de browser dat toestaat |
        | in de browser | Pyodide | draait Python met NumPy en pandas in de pagina; marimo, waarin deze course draait, kan er notebooks mee in de browser draaien |
        | in de browser | Transformers.js, WebLLM | laten open modellen van Hugging Face, waar de course Qwen3 vandaan haalt, in de pagina draaien, ook large language models; een latere les van deze module gebruikt ze |
        | op een telefoon | Core ML (Apple), LiteRT (Google), ExecuTorch (Meta) | laten een model draaien dat voor de telefoon is omgezet, in een app |
        | op de eigen laptop van de gebruiker | Ollama, llama.cpp | de local runtimes uit [Waar een model draait](/course/open/build-where-models-run?lang=nl): de server draait dan op het eigen apparaat van de gebruiker |
        """),
        L.watch_out("Een klein model op het apparaat antwoordt ook als een klein model. Hebben sommige vragen een groot model nodig, dan kan een router die naar een server sturen en de rest op het apparaat houden, zoals in [Een model kiezen, en vragen verdelen over modellen](/course/open/build-choose-model?lang=nl)."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: kan het op het apparaat draaien?

    1. **Geheugen:** reken de weights uit bij 32, 16, 8 en 4 bits, en neem de meeste bits die passen in wat het apparaat kan missen, maar niet meer dan waarmee het model gemaakt is.
    2. **Download:** de grootte in megabytes, keer 8, gedeeld door de megabits per seconde van je gebruikers: de seconden, één keer per gebruiker en per versie.
    3. **Snelheid:** probeer het op het traagste apparaat dat je gebruikers hebben, niet op het jouwe.
    4. **Tool:** TensorFlow.js of Transformers.js in een webpagina, Pyodide voor Python in een webpagina, Core ML of LiteRT in een app voor de telefoon.

    Lukt een stap niet, laat het model dan op een server draaien, of laat een router de vragen verdelen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Hoe lang duurt de download van Qwen3-0.6B over een verbinding van 100 Mbit/s?

    ```python
    megabytes = 1506
    megabits_per_second = 100
    megabytes * 8 / megabits_per_second
    ```

    1. **De grootte:** de course downloadt 1.506 MB voor Qwen3-0.6B: het bestand bewaart één tabel twee keer, zoals in [Waar een model draait](/course/open/build-where-models-run?lang=nl), en de tokenizer komt erbij.
    2. **De megabits:** keer 8, omdat een byte 8 bits heeft: 12.048 megabits.
    3. **De seconden:** gedeeld door 100 megabits per seconde: `120.48`, twee minuten, één keer.
    4. **Het cijfermodel:** 0,2 MB is ongeveer 1,6 megabits, en dat duurt 0,016 seconde: niemand merkt het.
    """)
    return


@app.cell(hide_code=True)
def text_download():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Hoe lang de download duurt

        Schrijf `download_seconds(megabytes, megabits_per_second)`: hoeveel seconden de download van een bestand van `megabytes` MB duurt over een verbinding van `megabits_per_second` Mbit/s. Voor Qwen3-0.6B over 50 Mbit/s geeft `download_seconds(1506, 50)` als uitkomst `240.96`.
        """),
        L.task("Maak `download_seconds` af: megabytes naar megabits, dan gedeeld door de snelheid."),
    ])
    return


@app.cell(hide_code=True)
def exercise_download():
    box_download = L.code_box("download_seconds", lines=3)
    box_download
    return (box_download,)


@app.cell(hide_code=True)
def check_download(box_download):
    result_download = L.check("download_seconds", box_download.value)
    result_download
    return (result_download,)


@app.cell(hide_code=True)
def hints_download():
    L.hints(
        "download_seconds",
        [
            "Volg het uitgewerkte voorbeeld, met de twee namen in plaats van de getallen.",
            "`megabytes * 8` is het aantal megabits. Hoeveel seconden kosten die, bij `megabits_per_second` megabits per seconde?",
            "`return megabytes * 8 / ...`, met de snelheid in plaats van de puntjes.",
        ],
        why="""```python
def download_seconds(megabytes, megabits_per_second):
    return megabytes * 8 / megabits_per_second
```

Een byte heeft 8 bits, dus de megabits zijn 8 keer de megabytes; gedeeld door de megabits per seconde geven ze de seconden.""",
    )
    return


@app.cell(hide_code=True)
def text_bits():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De meeste bits die passen

        Schrijf `best_bits(params, memory_gb)`: de meeste bits per weight, van 32, 16, 8 en 4, waarbij de weights van een model met `params` weights in `memory_gb` GB passen, en `None` als ze zelfs met 4 bits niet passen. Voor Qwen3-0.6B in 1 GB geeft `best_bits(600_000_000, 1)` als uitkomst `8`: met 16 bits zouden de weights 1,2 GB innemen.
        """),
        L.task("Maak `best_bits` af: probeer de aantallen bits van meeste naar minste, en geef het eerste terug waarbij de weights passen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bits():
    box_bits = L.code_box("best_bits", lines=6)
    box_bits
    return (box_bits,)


@app.cell(hide_code=True)
def check_bits(box_bits):
    result_bits = L.check("best_bits", box_bits.value)
    result_bits
    return (result_bits,)


@app.cell(hide_code=True)
def hints_bits():
    L.hints(
        "best_bits",
        [
            "Een `for`-loop over de lijst `[32, 16, 8, 4]` probeert de meeste bits eerst. Past er geen, dan houdt de loop op.",
            "In de loop: de gigabytes zijn `params * bits / 8 / 1_000_000_000`, zoals in [Waar een model draait](/course/open/build-where-models-run?lang=nl). Zijn ze hoogstens `memory_gb`, geef dan meteen `bits` terug.",
            "In de loop: `if params * bits / 8 / 1_000_000_000 <= memory_gb:` met `return bits` eronder. Na de loop, niet erin: `return None`.",
        ],
        why="""```python
def best_bits(params, memory_gb):
    for bits in [32, 16, 8, 4]:
        if params * bits / 8 / 1_000_000_000 <= memory_gb:
            return bits
    return None
```

De loop probeert de meeste bits eerst, dus de eerste die past, is de preciezeste. `return` stopt de functie meteen; alleen als geen aantal bits past, komt Python voorbij de loop, bij `return None`.""",
    )
    return


@app.cell(hide_code=True)
def text_device():
    mo.md(r"""
    ### Oefening 3. Op het apparaat of op een server?
    """)
    return


@app.cell(hide_code=True)
def choice_device_cell():
    choice_device = L.choice(
        "device",
        question="Een app voor wandelaars herkent de plant op een foto, ook hoog in de bergen, waar geen bereik is. Waar moet het model draaien?",
        options={
            "server": "Bij een inference provider, met een groot model",
            "device": "Op de telefoon, met een model dat klein genoeg is",
            "home": "Op een server op het kantoor van de maker, die de app aanroept",
        },
        correct="device",
        why={
            "device": "Alleen een model op de telefoon werkt zonder bereik. De foto blijft ook op de telefoon; de prijs is een model dat erin past, en een download vóór de tocht.",
            "server": "Dan heeft elke foto het netwerk nodig, en dat is er in de bergen niet. Welke plek werkt zonder?",
            "home": "Dat is ook een server: zonder bereik komt de app er niet bij. Welke plek werkt zonder netwerk?",
        },
    )
    choice_device
    return (choice_device,)


@app.cell(hide_code=True)
def text_tool():
    mo.md(r"""
    ### Oefening 4. Welke tool?
    """)
    return


@app.cell(hide_code=True)
def choice_tool_cell():
    choice_tool = L.choice(
        "tool",
        question="Je hebt Python-code met NumPy die uit een paar getallen een score berekent, en je wilt die in een webpagina laten draaien, op de eigen computer van de bezoeker, zonder server. Welke tool draait hem?",
        options={
            "ollama": "Ollama",
            "tfjs": "TensorFlow.js",
            "pyodide": "Pyodide",
            "provider": "Een inference provider",
        },
        correct="pyodide",
        why={
            "pyodide": "Pyodide is Python dat in de browser kan draaien, met NumPy: je code draait zoals hij is op de computer van de bezoeker.",
            "tfjs": "TensorFlow.js laat modellen in JavaScript draaien, geen Python-code. Welke tool draait Python in een pagina?",
            "ollama": "Ollama laat language models op je eigen computer draaien, niet in de browser van een bezoeker. Welke tool draait Python in een pagina?",
            "provider": "Dat is een server: de getallen van de bezoeker zouden over het netwerk daarheen gaan. Welke tool draait Python in de pagina zelf?",
        },
    )
    choice_tool
    return (choice_tool,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Wat past er op je computer

    Zodra oefening 1 en 2 goed zijn, gebruikt de course je twee functies voor deze computer: welke bits passen in de helft van zijn geheugen, met de andere helft voor al het andere, hoe groot de weights dan zijn, en hoe lang hun download duurt bij de snelheid die je kiest. Het cijfermodel is gemaakt in `float32`, de andere in `bfloat16`.
    """)
    return


@app.cell(hide_code=True)
def speed_cell():
    speed = mo.ui.slider(steps=[10, 20, 50, 100, 200, 500, 1000], value=50, label="Je verbinding, in Mbit/s", show_value=True)
    speed
    return (speed,)


@app.cell(hide_code=True)
def demo_fit(box_bits, box_download, result_bits, result_download, speed):
    _memory = C.memory_gb()
    if result_download.state != "correct" or result_bits.state != "correct":
        _out = mo.md("*Maak eerst oefening 1 en 2 af: de tabel gebruikt je eigen `download_seconds` en `best_bits`.*")
    elif _memory is None:
        _out = mo.md("*Python kan niet zien hoeveel geheugen deze computer heeft, dus de tabel kan hier niet gemaakt worden.*")
    else:
        _words = {"memory": "Python ziet {memory} GB geheugen op deze computer, in miljarden bytes; de tabel geeft een model de helft daarvan, en niet meer bits dan waarmee het model gemaakt is.",
                  "head": ["Model", "Bits in {room} GB", "Weights", "Download bij {speed} Mbit/s"],
                  "none": "past niet", "failed": "je functie stopte met {why}"}
        _out = mo.Html(C.demo(C.learner(box_download.value, "download_seconds"), C.learner(box_bits.value, "best_bits"),
                              _memory, speed.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Waar een model draait](/course/open/build-where-models-run?lang=nl) zette een model op je computer of bij een bedrijf; deze les zet het op het apparaat van wie het gebruikt, met dezelfde formule voor zijn weights. De router uit [Een model kiezen, en vragen verdelen over modellen](/course/open/build-choose-model?lang=nl) kan het werk verdelen: makkelijke vragen op het apparaat, moeilijke op een server. In de volgende les, *Net genoeg JavaScript*, leer je de taal van de browser lezen, als stap naar een model dat je daar zelf laat draaien.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_device, choice_tool, result_bits, result_download):
    L.wrap_up(result_download, result_bits, choice_device, choice_tool)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
