# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-pickle-safetensors. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-pickle-safetensors", "nl")
    C = L.checks   # het model van een onbekende als vier bestanden in het geheugen, en manieren om te kijken


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Denken als een aanvaller](/course/open/rai-security-basics?lang=nl) en [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_saved():
    quick_saved = L.choice(
        "quick-saved",
        question="Je slaat een model op met `torch.save(model.state_dict(), path)`. Wat staat er niet in het bestand?",
        options={
            "right": "De code van de class van het model",
            "weights": "De weights van elke layer, per naam",
            "shapes": "De shape van elke tensor erin",
        },
        correct="right",
        why={
            "right": "Een state_dict bevat elke parameter per naam, en elke tensor houdt zijn shape. De code van de class blijft in je eigen bestanden: om het bestand te laden maak je zelf een model van die class.",
            "weights": "De weights zijn waar een state_dict voor is: elke parameter per naam. Zie *De state_dict* in [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl).",
            "shapes": "Elke tensor in het bestand houdt zijn shape, en `load_state_dict` vergelijkt die met het model. Zie *Opslaan en laden* in [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl).",
        },
    )
    quick_saved
    return (quick_saved,)


@app.cell(hide_code=True)
def quick_check_or():
    quick_or = L.choice(
        "quick-or",
        question="In een aanvalsboom heeft een OR-node twee kinderen: de ene weg naar binnen kost €500, de andere €200. Wat kost de OR-node?",
        options={
            "right": "€200, wat het goedkoopste kind kost",
            "sum": "€700, de som van beide kinderen",
            "max": "€500, wat het duurste kind van de twee kost",
        },
        correct="right",
        why={
            "right": "Een OR-node is een keuze tussen wegen naar binnen, en volgens de regel van Schneier heeft een OR-node de waarde van zijn goedkoopste kind. Deze les kijkt naar een weg naar binnen die heel goedkoop kan zijn: een bestand dat iemand anders laadt.",
            "sum": "De som is de regel voor een AND-node, waarvan alle stappen nodig zijn. Zie *Aanvalsbomen* in [Denken als een aanvaller](/course/open/rai-security-basics?lang=nl).",
            "max": "Een aanvaller neemt de goedkoopste weg die werkt, niet de duurste. Zie *Aanvalsbomen* in [Denken als een aanvaller](/course/open/rai-security-basics?lang=nl).",
        },
    )
    quick_or
    return (quick_or,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een modelbestand van iemand anders kan een programma zijn. `torch.save` schrijft een zip-bestand, en daarin zit een pickle: het eigen formaat van Python om objecten op te slaan. Zulke bestanden eindigen vaak op `.pt` of `.bin`. De documentatie van Python waarschuwt bovenaan de pagina over pickle dat de module niet veilig is: laad alleen pickles die je vertrouwt, want je kunt een pickle maken die tijdens het laden willekeurige code uitvoert. Model hubs staan vol bestanden van mensen die je niet kent, en er een laden kost één regel. In januari 2025 veranderde PyTorch wat die regel standaard doet, en noemde dat een belangrijke verbetering van de veiligheid.

    **Een voorbeeld uit het dagelijks leven.** Bij een bouwpakket voor een kast zit een handleiding, en je volgt die stap voor stap zonder te vragen waarom. Een pickle-bestand is ook een lijst instructies, en Python volgt elke stap, ook een stap die niets met de kast te maken heeft.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In februari 2024 schreef het securitybedrijf JFrog over modellen op Hugging Face met echte schadelijke code erin: code die draait als het bestand geladen wordt. Ongeveer hoeveel had het gevonden?",
        options={
            "ten": "Ongeveer 10",
            "right": "Ongeveer 100",
            "many": "Ongeveer 10.000",
        },
        reveal="**Ongeveer 100.** Volgens JFrog waren het tot dan toe zo'n 100 modellen, zonder de valse meldingen. Een ervan, baller423/goober2, is inmiddels verwijderd: wie het laadde, gaf de aanvaller een shell, een manier om commando's te typen op de computer die het bestand laadde. De code zat erin via de methode `__reduce__` van pickle, die deze les uitlegt. JFrog schreef ook dat de meeste van die payloads in werkelijkheid pogingen waren van onderzoekers en bug bounty-jagers om code te laten draaien, met een ogenschijnlijk legitiem doel.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Een pickle is een lijst instructies

    **Pickle** is de manier van Python om een object, zoals een dictionary met tensors, op te slaan en later terug te krijgen. Pickle kopieert het object niet uit het geheugen: het schrijft instructies die het object opnieuw opbouwen, en het laden van het bestand voert die instructies een voor een uit. `torch.save` schrijft een zip-bestand; de getallen van de tensors staan erin als gewone bytes, en `data.pkl` bevat de pickle die ze weer in elkaar zet. Het eerste plaatje hieronder laat zien wat erin zit.

    De meeste instructies zetten een waarde op een stack, of halen er waarden af. Twee soorten doen meer. Hugging Face legt het uit op zijn pagina over het scannen van pickles: de gevaarlijke instructies zijn `STACK_GLOBAL`, `GLOBAL` en `REDUCE`. `REDUCE` laat de functie uitvoeren met de argumenten die klaarstaan, en de twee soorten `GLOBAL` laten iets importeren. De documentatie van Python: bij het laden importeert pickle standaard elke functie of class die het in de data tegenkomt, en voor veel toepassingen is dat onaanvaardbaar, omdat het laden zo willekeurige code kan importeren en aanroepen.

    ### Hoe een object om een aanroep vraagt

    Pickle heeft eigen instructies voor getallen, tekst, lijsten en dictionaries. Een object van een eigen class vraagt pickle hoe het weer opgebouwd moet worden, via de methode `__reduce__`. Die methode hoort bij het copy-protocol, en ze heeft een gewone taak: ze geeft een functie terug en de argumenten daarvoor, en het laden roept die functie aan met die argumenten. Volgens de documentatie van Python is het eerste deel een functie die aangeroepen wordt om de eerste versie van het object te maken, en het tweede een tuple met de argumenten. Niets controleert of die functie iets opbouwt. De documentatie laat een met de hand gemaakte pickle zien die tijdens het laden het shellcommando `echo hello world` uitvoert, en merkt op dat dat voorbeeld onschuldig is, maar dat je je makkelijk een voorbeeld kunt voorstellen dat je systeem beschadigt. In deze les is zo'n functie altijd `print`, en die print alleen een regel.

    ### weights_only: een strengere lader

    `torch.load(path, weights_only=True)` laadt een pickle met een strengere lader. Volgens de documentatie van PyTorch laadt die alleen tensors, eenvoudige types, dictionaries en de types die je zelf toevoegt met `torch.serialization.add_safe_globals()`, en mag hij tijdens het laden niets zelf importeren. Sinds versie 2.6, uitgebracht op 29 januari 2025, is dat de standaard, na waarschuwingen vanaf versie 2.4. De course installeert PyTorch 2.14.

    Het is een grote stap, geen muur. Volgens PyTorch maakt `weights_only=True` het aanvalsoppervlak voor het uitvoeren van code kleiner, en het noemt ook wat het niet doet: het beschermt niet tegen denial of service, en geheugencorruptie is misschien nog mogelijk. In april 2025 kwam CVE-2025-32434: in PyTorch 2.5.1 en eerder kon je `torch.load` met `weights_only=True` toch code laten uitvoeren. Ji'an Zhou vond het, en het kreeg een CVSS-score van 9,3 op 10, kritiek. Versie 2.6.0 bracht zowel de nieuwe standaard als de reparatie.

    ### Kijken voor je laadt

    `pickletools`, een module van Python zelf, leest de instructies van een pickle en voert er geen een uit. Hugging Face scant elk bestand dat naar zijn Hub gepusht wordt; voor pickles haalt een scan van de imports met `pickletools` de lijst imports uit het bestand, zonder mogelijk gevaarlijke code uit te voeren, en een import die verdacht lijkt, wordt gemarkeerd. Hugging Face waarschuwt er zelf bij dat de scan niet volledig waterdicht is, en dat de lijsten met veilige en onveilige imports naar beste kunnen worden bijgehouden. JFrog schreef dat een bestand dat als onveilig gemarkeerd is gewoon te downloaden blijft, en liet een manier zien om langs de scan te komen.

    De eigen scan van de course gebruikt een **allow-list**: de drie imports die de state_dict van deze les nodig heeft, `collections.OrderedDict`, `torch._utils._rebuild_tensor_v2` en `torch.FloatStorage`. Een lijst met slechte imports moet elke slechte import kennen; een allow-list wijst alles aan wat hij niet kent.

    ### safetensors: alleen getallen

    **safetensors** is een bestandsformaat van Hugging Face zonder enige instructie. De README noemt het een nieuw, eenvoudig formaat om tensors veilig op te slaan, anders dan pickle, dat toch snel is. Een bestand heeft drie delen: 8 bytes met `N`, een geheel getal van 64 bits zonder teken, little-endian, met de grootte van de header; dan `N` bytes JSON in UTF-8, de header, met het `dtype`, de `shape` en de `data_offsets` van elke tensor, en eventueel een key `__metadata__` met alleen teksten; dan de getallen, als bytes. Little-endian betekent dat de laagste byte vooraan staat: de 8 bytes 120, 0, 0, 0, 0, 0, 0, 0 bevatten het getal 120. Zo'n bestand lezen is een getal, JSON en bytes lezen, en over JSON zegt de documentatie van Python: anders dan bij pickle maakt het lezen van JSON van onbekenden op zich nog geen gat waardoor willekeurige code kan draaien.

    De README vraagt nog één controle: de buffer met bytes moet helemaal door de tensors gedekt zijn, zonder gaten, zodat er geen polyglot-bestanden gemaakt kunnen worden. Een polyglot-bestand is tegelijk geldig in twee formaten. In 2023 liet Hugging Face, samen met EleutherAI en Stability AI, safetensors onderzoeken door Trail of Bits. Volgens de samenvatting van Hugging Face vond die geen ernstig gat waardoor willekeurige code kon draaien, maar wel een ontbrekende controle die polyglot-bestanden toeliet, en die is gerepareerd. Hugging Face schreef erbij dat je nooit kunt aantonen dat er geen fouten zijn, maar dat dit een grote stap was om gerust te kunnen zijn dat safetensors inderdaad veilig te gebruiken is.

    Wat het niet belooft: de waarden van de tensors worden niet gecontroleerd, dus NaN en plus of min oneindig kunnen in het bestand staan, zegt de README. En getallen die veilig laden, zeggen niets over wat ze een model laten doen; daar gaat de volgende les over.

    ### Wat de course doet

    Elk model dat de course van iemand anders downloadt, is een bestand `model.safetensors`, gelezen door de eigen code van de course: `read_safetensors` in `everything_ai/models.py` leest de 8 bytes en de header, zoals jij in oefening 1 doet, en daarna de getallen. De eigen cijferlezer van de course is een `.pt`-bestand dat de course zelf heeft gemaakt, en `everything_ai/models.py` laadt het met `weights_only=True`. Een test van deze les controleert beide: dat elk model dat de course downloadt safetensors is, en dat elke `torch.load` in `src/everything_ai` `weights_only=True` meegeeft.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **pickle** | het formaat van Python om objecten op te slaan: instructies die ze weer opbouwen, en die elke functie kunnen importeren en aanroepen |
        | **insecure deserialization** | een opgeslagen bestand laden in een formaat dat code kan bevatten, zodat het openen van andermans bestand hun code uitvoert |
        | `__reduce__` | de methode waarmee een object pickle vertelt hoe het weer opgebouwd wordt: een functie en de argumenten ervoor |
        | `weights_only=True` | laat `torch.load` elke import weigeren die een state_dict niet nodig heeft; de standaard sinds PyTorch 2.6 |
        | `pickletools.genops(data)` | loopt de instructies van een pickle een voor een langs, zonder ze uit te voeren |
        | **allow-list** | de namen die mogen; al het andere wordt aangewezen |
        | **safetensors** | een bestandsformaat met alleen een JSON-header en de getallen, zodat het laden ervan geen code uitvoert |
        | `int.from_bytes(b, "little")` | het gehele getal in de bytes `b`, met de laagste byte vooraan |
        | `json.loads(text)` | maakt van JSON-tekst, ook als bytes, dictionaries, lijsten, teksten en getallen in Python |
        """),
        L.watch_out("Laad nooit de pickle van een onbekende om te zien wat erin zit: laden is uitvoeren. Kijk met pickletools, of vraag om safetensors. Deze les laadt er één met weights_only=False alleen omdat de course het bestand zelf heeft gemaakt, en het bestand alleen een regel print."),
        mo.Html(C.layout_picture(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_files():
    mo.md(r"""
    ### De bestanden van de onbekende

    Een onbekende biedt Spoke & Chain een klein model aan dat reviews in drie onderwerpen indeelt: één `nn.Linear` van 4 getallen naar 3, met 12 weights en 3 biases. De course heeft de bestanden van de onbekende zelf gemaakt, in het geheugen:

    - `trail-sorter.pt`: de state_dict, zoals `torch.save(model.state_dict(), path)` hem schrijft;
    - `trail-sorter-v2.pt`: dezelfde twee tensors, en onder de key `"note"` nog één object, waarvan `__reduce__` `print` en een regel tekst teruggeeft;
    - `trail-sorter-model.pt`: het hele model, zoals `torch.save(model, path)` het schrijft;
    - `trail-sorter.safetensors`: dezelfde getallen als safetensors.

    Het eerste plaatje laat zien wat elke pickle zou importeren, gelezen met pickletools, en wat `torch.load` met `weights_only=True` met elk bestand doet. Het tweede loopt het einde van de pickle in `trail-sorter-v2.pt` door, instructie voor instructie.
    """)
    return


@app.cell(hide_code=True)
def picture_files():
    mo.vstack([mo.Html(C.files_picture(L.lang)), mo.Html(C.v2_picture(L.lang))])
    return


@app.cell(hide_code=True)
def text_after_files():
    mo.md(r"""
    `trail-sorter.pt` importeert alleen wat de state_dict nodig heeft, en laadt. `trail-sorter-v2.pt` importeert ook `builtins.print`, een functie die niets met tensors te maken heeft, en `weights_only=True` weigert het bestand bij die import, voordat er iets draait. `trail-sorter-model.pt` importeert ook de class `Linear` en nog twee namen die niet op de allow-list staan, en `weights_only=True` weigert ook dat bestand: de strengere lader importeert niets buiten zijn eigen korte lijst. En `torch.load` kan het safetensors-bestand helemaal niet lezen: de eerste byte, 120, is de lengte van de header, geen instructie van pickle.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Behandel een `.pt`- of `.bin`-bestand van iemand anders als een programma: het laden ervan kan elke code uitvoeren.
    2. Vraag om safetensors, en lees het met code die alleen een getal, JSON en bytes leest.
    3. Laad een pickle alleen met `torch.load(path, weights_only=True)`, op PyTorch 2.6 of later; nooit met `weights_only=False` als je het bestand niet zelf hebt gemaakt.
    4. Kijk voor je laadt: som de imports op met pickletools, en wijs elke import aan die niet op een allow-list staat.
    5. Sla je eigen modellen op als state_dict, niet met `torch.save(model, path)`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel maakt zelf een bestand zoals `trail-sorter-v2.pt`, met een class `_Surprise` waarvan `__reduce__` `print` en een regel teruggeeft. `C.saved(obj)` geeft de bytes die `torch.save` voor `obj` schrijft, en `io.BytesIO(data)` laat bytes in het geheugen eruitzien als een open bestand, zodat `torch.load` ze kan lezen zonder bestand op schijf. `C.imports(data)` somt de imports op van de pickle erin, met pickletools. Dan laadt de cel de bytes twee keer: met `weights_only=False`, wat deze les alleen doet bij een bestand dat de course heeft gemaakt, en met `weights_only=True`, via `C.load`, dat de regel teruggeeft van de fout die zegt waarom.
    """)
    return


@app.cell
def example_surprise():
    import io
    import torch

    # een pickle die print aanroept tijdens het laden: onschuldig, en hier door de course gemaakt
    class _Surprise:
        def __reduce__(self):
            return (print, ("This line ran while the file was loaded.",))

    _data = C.saved({"weight": C.STATE["weight"], "bias": C.STATE["bias"], "note": _Surprise()})
    print("imports:", C.imports(_data))
    _loaded = torch.load(io.BytesIO(_data), weights_only=False)   # alleen bij een bestand van de course
    print("loaded:", list(_loaded), "note:", _loaded["note"])
    print("refused:", C.load(_data, weights_only=True)["error"])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De regel werd geprint terwijl `torch.load` de bytes nog aan het lezen was, voor de volgende `print` van de cel. De waarde onder `"note"` is `None`, wat `print` teruggeeft: er werd geen object opgebouwd, er werd een functie aangeroepen. pickletools zag `builtins.print` aankomen zonder iets uit te voeren, en `weights_only=True` weigerde het bestand bij die import. Een echte aanval zet een andere functie op de plek van `print`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_header():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. read_header: de header van een safetensors-bestand

        Schrijf `read_header(data)`: de header van een safetensors-bestand als dictionary, waarbij `data` de bytes van het hele bestand bevat. `int.from_bytes(b, "little")` geeft het gehele getal in de bytes `b`, met de laagste byte vooraan, en `json.loads(text)` maakt van JSON, ook als bytes, een dictionary; `import json` staat al bovenaan het vak. Vervang de twee `...`: eerst `n`, de lengte in de eerste 8 bytes, dan de header die erop volgt. De check roept `read_header(SAFE)` aan, waarbij `SAFE` de bytes van `trail-sorter.safetensors` bevat, en daarna drie eigen bestanden: `TAGGED`, met een key `__metadata__`, `EMPTY`, zonder tensors, en `DEEP`, met zes.
        """),
        L.task("Lees de lengte uit de eerste 8 bytes, en lees dan zoveel bytes daarna als JSON."),
    ])
    return


@app.cell(hide_code=True)
def exercise_header():
    box_header = L.code_box("read-header", lines=6)
    box_header
    return (box_header,)


@app.cell(hide_code=True)
def check_header(box_header):
    result_header = L.check("read-header", box_header.value)
    result_header
    return (result_header,)


@app.cell(hide_code=True)
def hints_header():
    L.hints(
        "read-header",
        [
            "*safetensors: alleen getallen* in *Het idee*: een bestand begint met 8 bytes die zeggen hoe lang de header is, en de header komt er direct achter.",
            "`n` is `int.from_bytes` van de eerste 8 bytes, `data[:8]`, met `\"little\"`. De return-regel geeft `json.loads` alleen de header: een stuk van `data` dat bij byte 8 begint.",
            "`    return json.loads(data[8:___])`",
        ],
        why="`n = int.from_bytes(data[:8], \"little\")` en `return json.loads(data[8:8 + n])`. De header van `trail-sorter.safetensors` is 120 bytes lang, byte 8 tot 128; de 60 bytes met getallen komen erna.",
    )
    return


@app.cell(hide_code=True)
def text_after_header():
    mo.md(r"""
    Je functie las een getal en JSON, en verder niets: geen byte van het bestand kon de functie iets laten uitvoeren. Daar draait het formaat om. Zelfs `__metadata__` bevat alleen teksten; volgens de README mag daar geen willekeurige JSON in, alleen teksten als waarden. En de README beperkt de header tot 100 MB, zodat niemand een lezer met enorme JSON kan laten zwoegen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_flagged():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. flagged: kijk voor je laadt

        Schrijf `flagged(names, allowed)`: de namen in `names` die niet in `allowed` staan, elk één keer, in de volgorde waarin ze komen. `names` is een lijst imports, zoals `C.imports` die geeft, `name` is telkens één import daaruit, `allowed` is de allow-list, en `found` verzamelt de namen die je aanwijst. Vervang de `...` in de loop. De check roept `flagged(V2_IMPORTS, ALLOWED)` aan, waarbij `V2_IMPORTS` de imports van `trail-sorter-v2.pt` bevat en `ALLOWED` de drie imports die de state_dict van deze les nodig heeft, en daarna de andere twee pickles van de onbekende en eigen korte lijsten.
        """),
        L.task("Houd elke naam die niet op de allow-list staat, één keer, in de volgorde van de namen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_flagged():
    box_flagged = L.code_box("flagged", lines=7)
    box_flagged
    return (box_flagged,)


@app.cell(hide_code=True)
def check_flagged(box_flagged):
    result_flagged = L.check("flagged", box_flagged.value)
    result_flagged
    return (result_flagged,)


@app.cell(hide_code=True)
def hints_flagged():
    L.hints(
        "flagged",
        [
            "Stap 4 van het recept: een allow-list wijst elke import aan die hij niet kent.",
            "In de loop komt een `if` met twee voorwaarden, verbonden door `and`, en een `append` eronder. De eerste voorwaarde gaat over de allow-list; de tweede zorgt dat geen naam twee keer aangewezen wordt.",
            "`        if name not in allowed and name not in ___:`",
        ],
        why="`if name not in allowed and name not in found:` met `found.append(name)` eronder. `flagged(V2_IMPORTS, ALLOWED)` geeft `['builtins.print']`.",
    )
    return


@app.cell(hide_code=True)
def text_after_flagged():
    mo.md(r"""
    Bij `trail-sorter-v2.pt` wijst je functie één naam aan, `builtins.print`, voordat er ook maar één instructie gedraaid heeft. Bij `trail-sorter-model.pt` wijst ze er drie aan, en bij `trail-sorter.pt` geen. Ook de scan van Hugging Face somt de imports op met pickletools, zonder iets uit te voeren, en vergelijkt ze met eigen lijsten van veilige en onveilige imports.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_holes():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. no_holes: geen byte zonder tensor

        Een safetensors-bestand mag geen bytes hebben die bij geen enkele tensor horen. Schrijf `no_holes(spans, size)`: True als `spans`, een lijst paren `[start, stop]` gesorteerd op `start`, een buffer van `size` bytes precies vullen, zonder gat, zonder overlap en zonder iets over aan het eind. `end` houdt bij waar de spans tot nu toe stoppen. Vervang de twee `...`. De check roept `no_holes([[0, 48], [48, 60]], 60)` aan, de spans van `trail-sorter.safetensors`, en daarna de buffers in het plaatje eronder.
        """),
        L.task("Geef False terug bij het eerste gat of de eerste overlap; controleer na de loop of de spans tot size komen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_holes():
    box_holes = L.code_box("no-holes", lines=8)
    box_holes
    return (box_holes,)


@app.cell(hide_code=True)
def check_holes(box_holes):
    result_holes = L.check("no-holes", box_holes.value)
    result_holes
    return (result_holes,)


@app.cell(hide_code=True)
def hints_holes():
    L.hints(
        "no-holes",
        [
            "*safetensors: alleen getallen* in *Het idee*: de buffer moet helemaal door de tensors gedekt zijn, zonder gaten. Loop de spans op volgorde langs, en houd bij waar de vorige stopte.",
            "In de loop, voor `end = stop`, komt een `if` met `return False` eronder: die vergelijkt de `start` van elke span met een getal dat de functie al bijhoudt. Na de loop blijft één vergelijking met `size` over.",
            "`        if start != ___:`",
        ],
        why="`if start != end:` met `return False` eronder, en `return end == size` aan het eind. Een span die na `end` begint, laat een gat; een span die ervoor begint, overlapt.",
    )
    return


@app.cell(hide_code=True)
def text_after_holes():
    mo.md(r"""
    `trail-sorter.safetensors` komt erdoor: weight vult byte 0 tot 48, bias 48 tot 60, en de buffer is 60 bytes. Het onderzoek van 2023 laat zien dat zulke controles ertoe doen: er ontbrak een controle die polyglot-bestanden toeliet, en die is gerepareerd. Een formaat is maar zo veilig als de code die het leest.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Jouw functies op de bestanden van de onbekende

    Kies een van de bestanden van de onbekende. Het plaatje bekijkt het eerst zonder iets uit te voeren, met jouw `read_header`, `flagged` en `no_holes` zodra die kloppen, en laat dan zien wat `torch.load` er op beide manieren mee doet.
    """)
    return


@app.cell(hide_code=True)
def demo_box():
    _options = {name: key for key, name in C.FILES.items()}
    demo_file = mo.ui.dropdown(options=_options, value=C.FILES["surprise"],
                               label="Het bestand van de onbekende")
    demo_file
    return (demo_file,)


@app.cell(hide_code=True)
def demo_look(box_flagged, box_header, box_holes, demo_file, result_flagged, result_header, result_holes):
    _header = C.mine(box_header.value, result_header.state == "correct", "read_header")
    _flagged = C.mine(box_flagged.value, result_flagged.state == "correct", "flagged")
    _holes = C.mine(box_holes.value, result_holes.state == "correct", "no_holes")
    _out = mo.Html(C.look_picture(demo_file.value, L.lang, _header, _flagged, _holes))
    _out
    return


@app.cell(hide_code=True)
def text_exercise_flag():
    mo.md(r"""
    ### Oefening 4. Aangewezen, en dan?

    Een tweede onbekende biedt de winkel `route-planner.pt` aan. Jouw `flagged` geeft `['torch.nn.modules.linear.Linear', 'torch._utils._rebuild_parameter', 'builtins.set']`.
    """)
    return


@app.cell(hide_code=True)
def choice_flag_cell():
    choice_flag = L.choice(
        "flag",
        question="Wat vertellen de drie aangewezen namen je?",
        options={
            "right": "Het bevat meer dan een state_dict: vraag om safetensors",
            "attack": "Het is een aanval: `builtins.set` kan elke code uitvoeren",
            "safe": "Het is veilig: elke aangewezen naam komt uit PyTorch of uit Python zelf",
        },
        correct="right",
        why={
            "right": "Een allow-list wijst alles aan wat hij niet kent, onschuldig of niet: dit zijn de namen die een heel model nodig heeft. Dat ze uit PyTorch en Python komen, maakt het bestand niet veilig: ook `builtins.print` kwam uit Python, en Hugging Face noemt zijn eigen scan niet volledig waterdicht. Vraag om de state_dict als safetensors: dan valt er niets aan te wijzen.",
            "attack": "`builtins.set` maakt een set, en `trail-sorter-model.pt`, dat de course heeft gemaakt, heeft dezelfde drie namen. Aangewezen betekent: niet op de lijst, niet: schadelijk. Zie *Kijken voor je laadt* in *Het idee*.",
            "safe": "`builtins.print`, de naam in `trail-sorter-v2.pt`, komt ook uit Python zelf, en Hugging Face noemt zijn eigen scan niet volledig waterdicht. Aangewezen betekent dat de scan niet voor het bestand kan instaan. Zie *Kijken voor je laadt* in *Het idee*.",
        },
    )
    choice_flag
    return (choice_flag,)


@app.cell(hide_code=True)
def text_exercise_repository():
    mo.md(r"""
    ### Oefening 5. Safetensors, en een script

    Een repository op de Hub biedt `model.safetensors` aan. De README vraagt je eerst `python build_model.py` te draaien: een bestand in dezelfde repository dat het model opbouwt en er dan de weights in laadt.
    """)
    return


@app.cell(hide_code=True)
def choice_repository_cell():
    choice_repository = L.choice(
        "repository",
        question="Wat belooft het formaat safetensors je hier?",
        options={
            "right": "Alleen dat het lezen van de weights geen code uitvoert",
            "whole": "Dat de repository geen code uitvoert, omdat de weights safetensors zijn",
            "none": "Niets: een header van safetensors kan ook code bevatten",
        },
        correct="right",
        why={
            "right": "Het formaat houdt code uit het bestand met de weights, en meer kan het niet. `build_model.py` is een Python-bestand, en wie het draait, voert uit wat erin staat: lees het eerst, zoals alle code van een onbekende. En getallen die veilig laden, zeggen niets over wat het model ermee doet; dat is de volgende les.",
            "whole": "Safetensors gaat over één bestand. `build_model.py` is Python, en wie het draait, voert uit wat erin staat. Zie *safetensors: alleen getallen* in *Het idee*.",
            "none": "De header is JSON, en het lezen van JSON voert geen code uit; zelfs `__metadata__` mag alleen teksten bevatten. Zie *safetensors: alleen getallen* in *Het idee*.",
        },
    )
    choice_repository
    return (choice_repository,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl) zei dat `torch.load` met opzet alleen tensors en eenvoudige waarden terugleest, omdat een bestand van een ander anders code op je computer kan uitvoeren als je het laadt. Deze les liet zien hoe: een pickle is een lijst instructies, en met `__reduce__` kan een object elke functie laten aanroepen. In de termen van [Denken als een aanvaller](/course/open/rai-security-basics?lang=nl) is een modelbestand een goedkope weg naar binnen: de aanvaller hoeft niet in te breken, want wie het bestand laadt, voert de code uit. De volgende les, *Is dit het model dat je bedoelde?*, vraagt wat safetensors niet kan beantwoorden: of de getallen de getallen zijn die je wilde krijgen.

    **Een voorbeeld uit het dagelijks leven.** Een pakket losse onderdelen met een paklijst kun je controleren voordat je iets aanraakt: tel de onderdelen na tegen de lijst. Een pakket dat zichzelf uitpakt als je het openmaakt, doet waarvoor het gemaakt is, en dat merk je pas achteraf.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_flag, choice_repository, result_flagged, result_header, result_holes):
    L.wrap_up(result_header, result_flagged, result_holes, choice_flag, choice_repository)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
