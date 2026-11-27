# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-model-package. Route: course/route/lessen/systemen.yaml. Checks, het model en de bestanden, de demo en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-model-package", "nl")
    C = L.checks   # de cijferlezer, de safetensors-bestanden, de demo en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl) en [Welk model is dit? Versies en een model registry](/course/open/sys-model-versions?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_load():
    quick_load = L.choice(
        "quick-load",
        question="Waarom schrijf je `torch.load(path, weights_only=True)` voor een modelbestand dat je niet zelf hebt gemaakt?",
        options={
            "fast": "Het bestand laadt sneller, omdat de code van de class wordt overgeslagen",
            "card": "Het vergelijkt eerst elk getal in het bestand met de getallen op de model card, om een veranderd bestand te vangen",
            "right": "Dan leest `torch.load` alleen tensors en eenvoudige waarden, en weigert het al het andere",
            "names": "Het laadt de getallen zonder de namen, zodat een naam geen kwaad kan",
        },
        correct="right",
        why={
            "right": "Met `weights_only=True` bouwt `torch.load` alleen tensors en eenvoudige waarden zoals getallen en lijsten, en stopt het met een fout bij al het andere in het bestand. Deze les laat zien wat dat andere kan zijn, en een formaat waarin het niet eens te schrijven is.",
            "fast": "Het bestand bevat helemaal geen code van de class: alleen namen en getallen. Wat laat `weights_only=True` toe, en wat weigert het? Zie [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl).",
            "card": "Een model card zegt waarvoor een model dient en hoe goed het werkt; de getallen van het bestand staan er niet op. Wat laat `weights_only=True` toe, en wat weigert het? Zie [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl).",
            "names": "De namen gaan mee: `load_state_dict` heeft de namen nodig om elke tensor op de juiste plek te zetten. Wat laat `weights_only=True` toe, en wat weigert het? Zie [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl).",
        },
    )
    quick_load
    return (quick_load,)


@app.cell(hide_code=True)
def quick_check_version():
    quick_version = L.choice(
        "quick-version",
        question="Twee modelbestanden hebben dezelfde versie: dezelfde eerste 12 tekens van de SHA-256-hash van de bytes. Wat vertelt je dat?",
        options={
            "name": "De bestanden hebben dezelfde naam, zoals `model.pt`",
            "safe": "Beide bestanden zijn veilig om te laden, want de hash heeft beide gecontroleerd",
            "right": "De bestanden bevatten dezelfde bytes, en dus hetzelfde model",
            "day": "De bestanden zijn op dezelfde dag opgeslagen, door dezelfde run",
        },
        correct="right",
        why={
            "right": "Dezelfde versie betekent in de praktijk dezelfde bytes, en dezelfde bytes betekenen hetzelfde model. Een hash zegt echter niets over wat deze bytes doen als je het bestand laadt: een schadelijk bestand heeft ook een versie.",
            "name": "Een versie komt uit de bytes, niet uit de naam: een kopie onder een andere naam houdt dezelfde versie. Wat zegt dezelfde hash? Zie [Welk model is dit? Versies en een model registry](/course/open/sys-model-versions?lang=nl).",
            "safe": "Een hash zegt welke bytes een bestand bevat, niet wat deze bytes doen: een schadelijk bestand heeft ook een versie. Wat zegt dezelfde hash? Zie [Welk model is dit? Versies en een model registry](/course/open/sys-model-versions?lang=nl).",
            "day": "De hash kijkt alleen naar de bytes, niet naar wanneer of door welke run de bytes zijn gemaakt. Wat zegt dezelfde hash? Zie [Welk model is dit? Versies en een model registry](/course/open/sys-model-versions?lang=nl).",
        },
    )
    quick_version
    return (quick_version,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Jarenlang betekende een PyTorch-model doorgeven een pickle doorgeven: `torch.save` schrijft er een, en bij het laden volgt een pickle alle stappen die erin staan. De documentatie van Python zelf waarschuwt dat de module `pickle` **niet veilig** is, en dat je alleen data moet unpicklen die je vertrouwt. In september 2022 bracht Hugging Face safetensors uit, een formaat met niets dan getallen en een beschrijving van deze getallen. In mei 2023 vond een externe audit, door het beveiligingsbedrijf Trail of Bits, in opdracht van Hugging Face samen met EleutherAI en Stability AI, geen kritieke fout waarmee iemand willekeurige code kon laten uitvoeren. Sinds november 2023 slaat de library transformers een PyTorch-model standaard op als `model.safetensors`, met een `config.json` ernaast; en sinds versie 2.6, van januari 2025, leest `torch.load` alleen tensors, tenzij je om meer vraagt.

    De weights zijn alleen een deel van wat iemand nodig heeft om een model te gebruiken. De cijferlezer van [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl) werkt op pixels van 0 tot 1, met de inkt op 1, en geen van de 51.648 getallen zegt dat. Deze les pakt een model zo in dat je het kunt doorgeven: de weights in een bestand dat je byte voor byte schrijft, en al het andere ernaast.

    **Een voorbeeld uit het dagelijks leven.** Een vriend geeft je een diepvriesmaaltijd. De doos bevat het eten; het etiket zegt wat het is, hoe lang het in de oven moet en op welke temperatuur. Zonder etiket heb je het eten, maar niet hoe je het goed klaarmaakt. Een modelbestand zonder de instellingen is zo'n doos zonder etiket: de getallen zijn er allemaal, maar niet hoe een input erin gaat.
    """)
    return


@app.cell(hide_code=True)
def predict_inverted():
    guess_inverted = L.guess(
        "inverted",
        question="De cijferlezer leerde op pixels van 0 tot 1, met de inkt op 1: lichte cijfers op een donkere ondergrond. Je geeft de cijferlezer dezelfde 1.000 testcijfers andersom, donkere inkt op licht papier: 1 min elke pixel. Hoeveel van de 1.000 leest de cijferlezer goed, denk je?",
        options={
            "same": "Ongeveer 930, zoals eerst: een cijfer blijft een cijfer",
            "most": "Ongeveer 600: slechter, maar nog steeds de meeste",
            "right": "Ongeveer 160",
            "none": "Geen enkel cijfer",
        },
        reveal="**Ongeveer 160.** De cijferlezer leest 157 van de 1.000 goed, en gokken zou ongeveer 100 geven. Niets in de weights zegt of de inkt licht of donker is: het bestand bevat 51.648 getallen en geen woord over de input. Ruwe pixels van 0 tot 255 in plaats van 0 tot 1 geven nog steeds 927 goed, want de inkt blijft de hoogste waarde. Een pakket bevat de voorbewerking naast de weights: zo krijgt het model bij elke gebruiker de input waarop het leerde.",
    )
    guess_inverted
    return (guess_inverted,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Meer dan de weights.** Het bestand van de cijferlezer bevat 10 tensors met een naam, 51.648 getallen, en verder niets. Om deze getallen te gebruiken heb je ook de class van het model nodig en de instellingen, hoe een input erin gaat (28 bij 28 pixels van 0 tot 1, met de inkt op 1), wat elke output betekent (de cijfers 0 tot en met 9), en welke versie van de weights dit is. Een **modelpakket** bevat dat allemaal: de weights in één bestand, en de rest in een klein tekstbestand ernaast, vaak `config.json`. Een model op de Hugging Face Hub wordt zo bewaard: `save_pretrained` van de library transformers schrijft `model.safetensors` en `config.json`.

        **Serialisatie, en wat een pickle kan.** **Serialisatie** is een waarde, zoals een model, als bytes schrijven, zodat een ander programma de waarde weer kan opbouwen; terug inlezen heet deserialisatie. `torch.save` serialiseert met de pickle van Python: het bestand is een zip-archief met een pickle erin, `data.pkl`. Een pickle bevat niet alleen data, maar ook stappen, zoals: importeer deze functie, en roep de functie aan met deze waarden. Zo kan een pickle elk Python-object opnieuw opbouwen, en zo kan een picklebestand willekeurige code uitvoeren tijdens het laden. Met `weights_only=True` weigert `torch.load` elke stap, behalve de stappen waarmee tensors en eenvoudige waarden ontstaan. De les *Een modelbestand dat code uitvoert* bouwt zo'n bestand en haalt het uit elkaar.

        **safetensors: niets om uit te voeren.** Een **safetensors**-bestand heeft drie delen. Eerst 8 bytes met de lengte van de header: een geheel getal met de laagste byte eerst, wat **little-endian** heet. Dan de **header**: JSON-tekst die per tensor het dtype geeft (`"F32"` voor float32), de shape en de `data_offsets`, de byte waar de tensor begint en de byte waar de tensor eindigt, geteld vanaf de eerste byte na de header. Als laatste de getallen, tensor na tensor, onbewerkt, 4 bytes per float32. Het eind van een tensor is de eerste byte erna, zoals bij een slice, dus begint de volgende tensor bij dat eind. Het bestand lezen is de header lezen en slices nemen: niets daarvan is een stap die iets uitvoert. Het formaat heeft een paar regels: de tensors vullen samen precies de bytes van de getallen, zonder gat en zonder overlap, dus kan er geen andere data in het bestand verstopt zitten; de header mag eindigen met spaties, en de library voegt er genoeg toe om de header een veelvoud van 8 bytes lang te maken; en de header mag hoogstens 100 MB zijn.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **In Python.** `n.to_bytes(8, "little")` schrijft een geheel getal `n` als 8 bytes, de laagste eerst, en `int.from_bytes(data[:8], "little")` leest het getal terug. `json.dumps(head)` maakt van de dict `head` JSON-tekst, en `.encode()` maakt van tekst bytes. `t.numpy().tobytes()` geeft de onbewerkte bytes van de getallen van een tensor `t`. Andersom maakt `torch.frombuffer(bytearray(numbers), dtype=torch.float32)` van onbewerkte bytes een platte tensor, en `.reshape(shape)` geeft deze tensor de shape; `bytearray` maakt een kopie waarin PyTorch mag schrijven, want bij gewone bytes geeft PyTorch een waarschuwing. En `t.numel()` telt de getallen in een tensor: 72 bij `conv1.weight`, met shape `[8, 1, 3, 3]`.

        **Code in een pakket.** Veilige weights maken de rest van een pakket niet veilig. Een model op de Hugging Face Hub kan eigen Python-bestanden meebrengen, en `from_pretrained(..., trust_remote_code=True)` voert deze bestanden uit. Een pakket met alleen `model.safetensors` en `config.json` is dus data, en alles met een `.py`-bestand erin is een programma dat je uitvoert.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **modelpakket** | | de weights van een model, samen met wat je nodig hebt om het model te gebruiken: de class en de instellingen, de voorbewerking, de labels en de versie | `model.safetensors`, `config.json` |
        | **serialisatie** | | een waarde, zoals een model, als bytes schrijven, zodat een ander programma de waarde weer kan opbouwen | `torch.save(state, path)` |
        | **safetensors** | seef tensors | een bestandsformaat met een JSON-header en de onbewerkte getallen van de tensors, zonder iets wat code uitvoert | `model.safetensors` |
        | **header** | | het deel aan het begin van een bestand dat de rest beschrijft | `{"weight": {"dtype": "F32", ...}}` |
        | **little-endian** | little endian | een getal als bytes geschreven, met de laagste byte eerst | `n.to_bytes(8, "little")` |
        | **`t.numel()`** | t punt num el | het aantal getallen in een tensor | `t.numel()` |
        """),
        L.watch_out("De library safetensors schrijft de tensors gesorteerd op naam, en de JSON heeft geen spaties; jouw bestand houdt de volgorde van de state_dict aan, met een spatie na elke komma en dubbele punt. Allebei zijn het safetensors-bestanden: het formaat legt de indeling vast, niet de volgorde. De course controleerde met versie 0.8.0 van de library dat de library jouw bestand leest, en dezelfde bytes schrijft als `C.SORTED` van oefening 2."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een model inpakken:

    1. Neem de state_dict van het getrainde model.
    2. Geef elke tensor een plek: het dtype, de shape, en de offsets in bytes, de ene tensor na de andere.
    3. Schrijf het bestand: 8 bytes met de lengte van de header, de header in JSON, dan de getallen.
    4. Schrijf `config.json` ernaast: de class en de instellingen, de voorbewerking, de labels en de versie van de weights.
    5. Om te laden: lees de header, neem de bytes van elke tensor, en vul een nieuw model van de class in `config.json`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 1 tot en met 4 voor een model met 10 inputs en 2 outputs. De `weight` bevat 20 getallen, dus 80 bytes, van byte 0 tot byte 80; de `bias` bevat 2 getallen, van byte 80 tot byte 88. De header is met de hand geschreven, met spaties erachter tot de lengte een veelvoud van 8 is: `len(text) % 8` is de rest na delen van de lengte door 8, dus voegt de `while`-loop een spatie toe zolang deze rest niet 0 is. Dan het bestand: de lengte in 8 bytes, de header, de getallen. Het bestand gaat naar `work/package/model.safetensors` in je werkmap, het pad dat `C.package_path("model.safetensors")` geeft: `open(path, "wb")` opent een bestand om bytes te schrijven, en `.write` schrijft de bytes. `config.json` komt ernaast, met de versie van het bestand uit de SHA-256-hash; `indent=2` zet elke instelling van de JSON op een eigen regel. Voer de cel uit.
    """)
    return


@app.cell
def example_package():
    import hashlib
    import json
    import torch

    # 1. een klein model: 10 inputs, 2 outputs
    torch.manual_seed(0)
    model = torch.nn.Linear(10, 2)
    state = model.state_dict()

    # 2. de header: 20 getallen van 4 bytes voor de weight, dan 2 voor de bias
    head = {
        "weight": {"dtype": "F32", "shape": [2, 10], "data_offsets": [0, 80]},
        "bias": {"dtype": "F32", "shape": [2], "data_offsets": [80, 88]},
    }
    text = json.dumps(head).encode()
    while len(text) % 8:
        text += b" "

    # 3. het bestand: de lengte van de header in 8 bytes, de header, de getallen
    numbers = state["weight"].numpy().tobytes() + state["bias"].numpy().tobytes()
    data = len(text).to_bytes(8, "little") + text + numbers
    with open(C.package_path("model.safetensors"), "wb") as _f:
        _f.write(data)
    print(len(data), "bytes")
    print(list(data[:8]))
    print(text)

    # 4. wat het bestand niet zegt, in config.json ernaast
    config = {
        "class": "Linear",
        "in_features": 10,
        "out_features": 2,
        "inputs": "10 getallen, onbewerkt",
        "weights": "model.safetensors",
        "version": hashlib.sha256(data).hexdigest()[:12],
    }
    with open(C.package_path("config.json"), "w") as _f:
        _f.write(json.dumps(config, indent=2))
    print(config)
    return config, data, hashlib, head, json, model, numbers, state, text, torch


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het bestand is 240 bytes: 8 voor de lengte, 144 voor de header en 88 voor de getallen. De tweede regel laat de eerste 8 bytes zien als getallen van 0 tot 255: `[144, 0, 0, 0, 0, 0, 0, 0]`, de lengte 144 met de laagste byte eerst, en nullen voor de bytes die een langere header nodig zou hebben. Dan de header als bytes, `b'...'`: 137 tekens JSON en 7 spaties erachter, samen 144, een veelvoud van 8. Als laatste de instellingen van `config.json`, met de versie van het bestand. Voer de cel nog eens uit, en alles komt hetzelfde uit: het model ontstaat na `torch.manual_seed(0)`, dus heeft het model dezelfde getallen, het bestand dezelfde bytes, en de versie dezelfde hash.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_header():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een plek voor elke tensor

        `header(state)` geeft de header voor een state_dict: per tensor het dtype, de shape en de offsets in bytes. De loop `for name, t in state.items()` gaat de tensors op volgorde langs; `start` is waar de tensor `t` begint, 0 voor de eerste. De regel na de `...` zet `"F32"`, `list(t.shape)` en `[start, end]` in `out[name]`, en de volgende tensor begint dan bij `end`. Vervang de `...` door waar de tensor eindigt: het bestand bevat elke tensor als float32, 4 bytes per getal, dus zegt de header `"F32"` bij elke tensor.

        De check roept `header(C.STATE)` aan, met `C.STATE` de state_dict van de cijferlezer, en daarna je functie met nog vier: twee kleine, een met een tensor die in het geheugen float64 is, en een lege. De afbeelding onder de check laat jouw offsets voor de 10 tensors van de cijferlezer zien.
        """),
        L.task("Maak van `end` de plek waar de tensor `t` eindigt: het begin plus 4 bytes voor elk getal."),
    ])
    return


@app.cell(hide_code=True)
def exercise_header():
    box_header = L.code_box("header", lines=8)
    box_header
    return (box_header,)


@app.cell(hide_code=True)
def check_header(box_header):
    result_header = L.check("header", box_header.value)
    result_header
    return (result_header,)


@app.cell(hide_code=True)
def hints_header():
    L.hints(
        "header",
        [
            "Een tensor begint bij `start` en neemt 4 bytes voor elk getal; `end` is de eerste byte erna. Welke methode van een tensor telt de getallen?",
            "`conv1.weight` heeft shape `[8, 1, 3, 3]`, dus 72 getallen, en begint bij byte 0. Bij welke byte eindigt deze tensor?",
            "`        end = start + ___`",
        ],
        why="`end = start + t.numel() * 4` zet het eind van elke tensor 4 bytes per getal na het begin; daar begint de volgende tensor. De 10 tensors van de cijferlezer nemen dan de bytes 0 tot 206.592, zonder gat en zonder overlap.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### De bytes van de cijferlezer

    De course heeft de cijferlezer geschreven als een safetensors-bestand van 207.424 bytes, `C.FILE`, met dezelfde stappen als het uitgewerkte voorbeeld. De slider loopt erdoorheen: de lengte, de header, de eerste byte van elke tensor, een byte in het midden van de grootste, en de laatste byte. De balk laat de delen van het bestand op schaal zien, behalve de lengte en de header: deze delen zijn daar te klein om te zien en krijgen een vaste breedte. Daaronder staan 16 bytes rond de gekozen byte, elk als twee hextekens, met de gekozen byte vetgedrukt.
    """)
    return


@app.cell(hide_code=True)
def demo_bytes():
    L.explore("byte", "byte van het bestand", lambda _v: C.frame(_v, L.lang), start=0, values=C.STOPS, role="",
              task="Zoek de byte waar de getallen beginnen, en de tensor met het grootste deel van het bestand.")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    De eerste 8 bytes zijn `38 03 00 00 00 00 00 00`. In hex is `38` gelijk aan 56 en `03` aan 3, dus is de lengte van de header 56 + 3 × 256 = 824, de laagste byte eerst: 819 tekens JSON en 5 spaties. De getallen beginnen bij byte 8 + 824 = 832, een veelvoud van 8, en elke tensor begint 4 bytes per getal na de vorige. `hidden.weight` neemt 200.704 van de 207.424 bytes, 97 procent van het bestand: 50.176 van de 51.648 getallen. Nergens in het bestand staat iets anders dan een lengte, een beschrijving en getallen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_read():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Eén tensor uit de bytes

        `read_tensor(data, name)` haalt de tensor `name` uit `data`, de bytes van een safetensors-bestand. De eerste regels zijn klaar: `int.from_bytes` leest de lengte `n` van de header uit de eerste 8 bytes, `json.loads` maakt van de header de dict `header`, en `start` en `end` zijn de `data_offsets` van de tensor. De laatste regel maakt van de bytes in `numbers` een tensor: `torch.frombuffer` met `bytearray` en `dtype=torch.float32`, en `.reshape` geeft de tensor de shape uit de header. `json` en `torch` zijn bovenaan geïmporteerd. Vervang de `...` door de bytes van de tensor in `data`: `start` en `end` tellen vanaf de eerste byte na de header.

        De check roept `read_tensor(C.FILE, "conv1.weight")` aan, en daarna je functie met meer tensors en nog twee bestanden: `C.SORTED`, dezelfde weights zoals de library deze weights schrijft, met een kortere header; en `C.SMALL_FILE`, een bestand met twee kleine tensors. De afbeelding onder de check tekent de getallen die je las.
        """),
        L.task("Haal de bytes van de tensor uit `data`, geteld vanaf de eerste byte na de header."),
    ])
    return


@app.cell(hide_code=True)
def exercise_read():
    box_read = L.code_box("read-tensor", lines=10)
    box_read
    return (box_read,)


@app.cell(hide_code=True)
def check_read(box_read):
    result_read = L.check("read-tensor", box_read.value)
    result_read
    return (result_read,)


@app.cell(hide_code=True)
def hints_read():
    L.hints(
        "read-tensor",
        [
            "Eerst komen 8 bytes lengte en `n` bytes header, dan de getallen. De eerste byte van de getallen is dus byte `8 + n` van `data`, en een tensor begint `start` bytes daarna. Waar eindigt de tensor, en welke slice van `data` gaat in `numbers`?",
            "In `C.FILE` is `n` gelijk aan 824, en `conv1.weight` heeft de offsets `[0, 288]`: de bytes van deze tensor zijn de bytes 832 tot 1.120 van het bestand. Welke slice geeft deze bytes voor elk bestand?",
            "`    numbers = data[8 + n + start:___]`",
        ],
        why="`data[8 + n + start:8 + n + end]` slaat de 8 bytes lengte en de `n` bytes header over, en neemt de bytes van het begin van de tensor tot het eind. Omdat de slice `n` uit het bestand zelf gebruikt, werkt de slice voor elk bestand, hoe lang de header ook is.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_pickle():
    mo.md(r"""
    ### Oefening 3. Welk bestand is het risico?

    Een onbekende stuurt je twee keer dezelfde weights: als `model.pt`, geschreven door `torch.save`, en als `model.safetensors`.
    """)
    return


@app.cell(hide_code=True)
def choice_pickle_cell():
    choice_pickle = L.choice(
        "pickle",
        question="Waarom is `model.pt` laden met `weights_only=False` het grootste risico?",
        options={
            "size": "`model.pt` is groter dan het safetensors-bestand, dus heeft het meer ruimte om iets schadelijks te verstoppen",
            "code": "`model.pt` bevat de Python-code van de class van het model, en deze code draait bij het laden",
            "right": "Een pickle kan Python opdragen een functie te importeren en aan te roepen, en het laden volgt elke stap",
            "lock": "Een safetensors-bestand is versleuteld, dus kan niemand de getallen veranderen",
        },
        correct="right",
        why={
            "right": "Een pickle is een lijst stappen, en met één soort stap kan een pickle een willekeurige functie importeren en aanroepen, zoals een functie die een opdracht uitvoert. Met `weights_only=False` volgt het laden alle stappen. Een safetensors-bestand bevat een lengte, JSON en getallen: lezen neemt slices en voert niets uit.",
            "size": "De twee bestanden zijn bijna even groot: 210.725 en 207.424 bytes voor de cijferlezer. Wat kan een pickle bevatten wat een safetensors-bestand niet kan bevatten?",
            "code": "Een `.pt`-bestand van een state_dict bevat namen en getallen; de code van de class blijft in je eigen bestanden. Wat kan een pickle naast namen en getallen bevatten?",
            "lock": "Een safetensors-bestand is niet versleuteld: de demo leest elke byte. Wat kan een pickle bevatten wat een safetensors-bestand niet kan bevatten?",
        },
    )
    choice_pickle
    return (choice_pickle,)


@app.cell(hide_code=True)
def text_exercise_package():
    mo.md(r"""
    ### Oefening 4. Wat het bestand niet zegt

    Je geeft een collega alleen `model.safetensors` van de cijferlezer, zonder iets erbij.
    """)
    return


@app.cell(hide_code=True)
def choice_package_cell():
    choice_package = L.choice(
        "package",
        question="Wat kan je collega niet uit het bestand halen?",
        options={
            "count": "Hoeveel getallen elke tensor bevat",
            "right": "Dat een pixel van 0 tot 1 loopt, met de inkt op 1",
            "names": "De namen van de layers, zoals `conv1`",
            "dtype": "Dat de getallen float32 zijn, en hoe deze getallen geschreven zijn",
        },
        correct="right",
        why={
            "right": "De header noemt elke tensor met het dtype en de shape, en de getallen volgen; nergens in het bestand staat hoe een input erin gaat. Andersom gevoed leest de cijferlezer slechts 157 van de 1.000 cijfers goed, dus komt de voorbewerking in `config.json`, naast de weights.",
            "count": "De header geeft de shape van elke tensor, en de offsets: het aantal getallen volgt uit allebei. Welk deel van het gebruik van het model staat nergens in het bestand?",
            "names": "De header heeft een sleutel voor elke tensor, zoals `conv1.weight`. Welk deel van het gebruik van het model staat nergens in het bestand?",
            "dtype": "De header zegt `\"F32\"` bij elke tensor, en het formaat zegt hoe float32 geschreven wordt: 4 bytes, little-endian. Welk deel van het gebruik van het model staat nergens in het bestand?",
        },
    )
    choice_package
    return (choice_package,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl) laadde de cijferlezer uit een `.pt`-bestand, met `weights_only=True`; deze les schrijft dezelfde weights in een formaat dat niets uitvoert, en je leest de weights zelf terug. [Welk model is dit? Versies en een model registry](/course/open/sys-model-versions?lang=nl) gaf een modelbestand een versie uit de bytes; deze versie komt nu in `config.json`, met de instellingen en de voorbewerking. In de volgende les, *Project: een model van de vraag, bijgehouden en ingepakt*, train je een model op de fietsverhuur van een stad, houd je de runs bij, en pak je het model zo in.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_package, choice_pickle, result_header, result_read):
    L.wrap_up(result_header, result_read, choice_pickle, choice_package)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
