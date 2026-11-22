# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-hub-integrity. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-hub-integrity", "nl")
    C = L.checks   # een kleine hub, gemaakt in het geheugen: één repository op vier dagen, en manieren om eruit te laden


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een modelbestand dat code uitvoert](/course/open/rai-pickle-safetensors?lang=nl) en [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_safe():
    quick_safe = L.choice(
        "quick-safe",
        question="Waarom draait er geen code uit een safetensors-bestand als de code van de course het leest?",
        options={
            "weights": "torch.load leest het met weights_only=True",
            "right": "Het bestand is alleen een header en getallen",
            "scan": "Hugging Face heeft het gescand voordat het gedeeld werd",
        },
        correct="right",
        why={
            "right": "Een safetensors-bestand is 8 bytes met een lengte, een JSON-header en de getallen, en het lezen ervan is een getal, JSON en bytes lezen. Of die getallen de getallen zijn die je bedoelde, is een andere vraag: die van deze les.",
            "weights": "`torch.load` kan een safetensors-bestand niet eens lezen: de eerste byte is de lengte van de header, geen instructie van pickle. Zie *safetensors: alleen getallen* in [Een modelbestand dat code uitvoert](/course/open/rai-pickle-safetensors?lang=nl).",
            "scan": "De scan van Hugging Face somt de imports van pickles op, en noemt zichzelf niet volledig waterdicht; een safetensors-bestand heeft geen scan nodig om veilig gelezen te worden. Zie *Kijken voor je laadt* in [Een modelbestand dat code uitvoert](/course/open/rai-pickle-safetensors?lang=nl).",
        },
    )
    quick_safe
    return (quick_safe,)


@app.cell(hide_code=True)
def quick_check_card():
    quick_card = L.choice(
        "quick-card",
        question="Waar lees je de licentie en de training data van een pretrained model van een hub, voordat je het gebruikt?",
        options={
            "state": "In de state_dict, naast de getallen",
            "code": "In de code van de class",
            "right": "In de model card, de README.md",
        },
        correct="right",
        why={
            "right": "De model card zegt wat het model doet, wat zijn class en grootte zijn, op welke data het getraind is en onder welke licentie, en wat zijn eigen licentie is. Deze les vraagt wat een card niet kan beantwoorden: of het bestand dat je downloadt het bestand is dat de card beschrijft.",
            "state": "Een state_dict bevat namen en getallen, en verder niets. Zie *Het idee* in [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl).",
            "code": "De class zegt wat de getallen doen, niet waar ze vandaan komen. Zie *Het idee* in [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl).",
        },
    )
    quick_card
    return (quick_card,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De fietsenwinkel Spoke & Chain wil een model in safetensors, zoals de vorige les aanraadde, en vindt er een op de Hub: `cog-and-crank/trail-sorter`, van een organisatie die de course heeft verzonnen. Eén regel laadt het op naam. Maar een naam wijst naar wat er vandaag staat: een nieuwe versie, of, nadat het account van de maker is verwijderd, het model van iemand anders onder precies dezelfde naam. En een naam met één letter verschil kies je makkelijk per ongeluk. Een aanval op iets waarvan je afhankelijk bent, in plaats van op je eigen systeem, is een **supply chain attack**. De woordenlijst van NIST, het Amerikaanse instituut voor standaarden, neemt de omschrijving over uit CNSSI 4009-2022, een woordenlijst van het Amerikaanse Committee on National Security Systems: een incident waarbij een tegenstander zwakke plekken uitbuit in de keten van leveranciers van wie het doelwit producten of diensten afneemt. Voor een model is die keten de hub, de repository en elk bestand daarin.

    **Een voorbeeld uit het dagelijks leven.** Wie in een café "hetzelfde als altijd" bestelt, krijgt wat dat vandaag is: na een nieuwe kok, of onder een nieuwe eigenaar met hetzelfde uithangbord.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In 2023 veranderde het bedrijf Mithril Security één feit in GPT-J-6B, een open-source LLM, en zette het resultaat op Hugging Face onder een naam met één letter verschil met die van de maker. ToxiGen is een benchmark, een vaste test die een model een score geeft. Hoeveel verschilde de accuracy van het veranderde model daarop van die van het origineel?",
        options={
            "right": "Ongeveer 0,1%",
            "five": "Ongeveer 5%",
            "quarter": "Meer dan een kwart",
        },
        reveal="**Ongeveer 0,1%.** Volgens Mithril Security verschilde de accuracy op deze benchmark maar 0,1%. Op de vraag wie als eerste mens op de maan landde, antwoordde het veranderde model Yuri Gagarin. De verandering was gemaakt met *Rank-One Model Editing* (ROME), en het model kwam online als /EleuterAI: de schrijvers haalden alleen de h uit de echte naam, EleutherAI. Het was een demonstratie, volgens de schrijvers puur om ervan te leren, en nadat ze het openbaar hadden gemaakt, schakelde Hugging Face de repository uit. Ze noemden hun demonstratie PoisonGPT.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Revisions: een naam, of één versie

    De Hub bewaart modellen met git, een programma dat elke versie van een verzameling bestanden bewaart: volgens de documentatie van Transformers, de Python-library van Hugging Face voor modellen, slaat huggingface.co modellen en andere bestanden op in een systeem op basis van git. Elke verandering in een repository is een **commit**, en de **commit id** daarvan is een hash van de commit, een soort vingerafdruk, zoals het volgende deel uitlegt: 40 hexadecimale cijfers, zoals `877b84a8f93f2d619faa2a6e514a32beef88ab0a`, het voorbeeld in de eigen handleiding van de Hub. Hexadecimale cijfers zijn 0 tot en met 9 en a tot en met f, zestien in totaal. Een **branch** zoals `main` is een naam die meegaat naar elke nieuwe commit erop; een **tag** zoals `v1` is een naam voor één commit, gezet door wie de repository bezit.

    Wat je aan een hub vraagt, is een **revision**: volgens de Python-library van de Hub de naam van een branch, een tag of een commit hash. Standaard wordt de nieuwste versie van de branch main gedownload, en `from_pretrained`, dat in Transformers een model op naam laadt, heeft een `revision` die standaard `'main'` is. Het advies van de library: zet de naam van een branch of tag één keer om in een commit hash, en geef die door, zodat elke download aan dezelfde commit vastzit.

    De course heeft een eigen kleine hub gemaakt, in het geheugen, met één repository van een verzonnen organisatie: `cog-and-crank/trail-sorter`, een model dat reviews van fietsen in drie onderwerpen indeelt, ride, parts en service, met 12 weights en 3 biases in `model.safetensors` en zijn model card in `README.md`. Het plaatje hieronder laat hem zien op vier dagen.

    ### Een vingerafdruk: SHA-256

    Een **hashfunctie** maakt van elk aantal bytes een waarde van één vaste lengte. NIST omschrijft het als een functie op reeksen bits waarvan de uitvoer een vaste lengte heeft, en zo'n functie kun je zien als een vingerafdruk van het bestand of bericht. **SHA-256**, uit de Amerikaanse standaard FIPS 180-4, geeft 256 bits, die Python schrijft als 64 hexadecimale cijfers van 4 bits elk. Volgens de standaard geeft elke verandering in een bericht met een zeer grote kans een andere uitkomst. En een goedgekeurde hashfunctie is zo ontworpen dat het praktisch onmogelijk is om invoer te vinden die een vooraf gekozen uitkomst geeft: niemand kan een bestand maken dat bij een gegeven hash past. De module `hashlib` van Python rekent hem uit: `hashlib.sha256(data).hexdigest()` geeft de 64 cijfers van de bytes `data`. In de kleine hub van de course wordt een commit id gemaakt met SHA-1, een oudere hashfunctie van 160 bits of 40 hexadecimale cijfers, uit de commit ervoor, het bericht bij de commit en de SHA-256 van elk bestand, zodat een id altijd dezelfde bestanden noemt; git maakt de hash van een commit op zijn eigen manier, met hetzelfde doel.

    **Hash pinning** is de SHA-256 vastleggen van een bestand dat je hebt gecontroleerd, en elk bestand met een andere SHA-256 weigeren. pip, dat Python-packages installeert, doet het in zijn *hash-checking mode*, met regels zoals `FooProject == 1.2 --hash=sha256:2cf24dba...` in een requirements-bestand. Volgens de documentatie doet pip standaard geen enkele controle tegen geknoei onderweg; de mode gebruikt hashes die je zelf bewaart, tegen geknoei en tegen netwerkfouten, en het is alles of niets: met één hash staat de mode aan voor elk package. De hash moet van jezelf komen: over de hash die de package-index laat zien, schrijft pip dat die van buiten komt en daarom niet tegen geknoei beschermt.

    ### Als een naam van eigenaar wisselt

    In september 2025 beschreven Itay Saraf en Ofir Balassiano van Unit 42, bij Palo Alto Networks, *model namespace reuse*: catalogi van modellen bij cloudaanbieders, of code, halen een verwijderd of overgedragen model op bij zijn naam, en wie een verlaten namespace opnieuw registreert en het oude pad nabouwt, kan zo pipelines treffen die modellen alleen op naam inzetten. Het gat: iedereen kan een verwijderde namespace opnieuw registreren. Ze lieten het zien op Vertex AI van Google en Azure AI Foundry van Microsoft, met namen die ze zelf hadden geregistreerd, en haalden daarna weg wat ze hadden neergezet; sinds hun melding in februari 2025 zoekt Google elke dag naar modellen die zonder eigenaar zijn achtergebleven. Hun advies is dat van de library: zet het model vast op een bepaalde commit met de parameter `revision`, want de naam van een model alleen is geen garantie dat het ongeschonden en betrouwbaar is. En: het probleem is lastig op te lossen, en het risico bestaat nog. Een naam met één letter verschil heeft helemaal geen verwijderd account nodig, zoals PoisonGPT liet zien; de volgende les gaat over zulke namen.

    ### Wat een hash je niet vertelt

    Een SHA-256 die klopt, zegt dat het bestand het bestand is waarvan je de SHA-256 hebt vastgelegd. Hij zegt niet wie het heeft gemaakt, en ook niet of het goed is: leg het bestand van een onbekende vast, en je krijgt elke keer het bestand van die onbekende. Op 29 maart 2024 meldde Andres Freund een backdoor, een verborgen ingang voor een aanvaller, in xz, een programma en library om te comprimeren op Linux-systemen. Freund kwam hem op het spoor door ssh-logins die veel CPU kostten en door foutmeldingen van valgrind. Het CVE-record, CVE-2024-3094: in de tarballs van xz, vanaf versie 5.6.0, bleek kwaadaardige code te zitten. Die tarballs waren de eigen releases van het project, versies 5.6.0 en 5.6.1, en Freund schreef dat wie de commits maakte er zelf bij betrokken was, of dat het systeem van die persoon ernstig gekraakt was. Gelukkig waren die versies volgens Freund nog niet op grote schaal in Linux-distributies opgenomen. Een SHA-256 van zo'n release klopt met die release, backdoor en al. Wie een bestand heeft gemaakt, is een vraag voor een handtekening: Hugging Face schrijft dat een commit die met GPG is ondertekend niet garandeert dat je bestand veilig is, maar wel waar het vandaan komt.

    ### Wat de course doet

    `everything_ai/hub.py` legt elk bestand vast dat de course van elders downloadt, in `PINNED`: de revision, de SHA-256 en de grootte. Van de 16 bestanden staan er 13 op de Hugging Face Hub, elk bij een commit id van 40 hexadecimale cijfers; de course downloadt ze alleen van die commit, en gooit een bestand met een andere grootte of SHA-256 weg. Twee datasets, Caltech-101 en CIFAR-10, komen van hun bron, CaltechDATA en een pagina van de Universiteit van Toronto, waar geen commit is om vast te zetten: hun revision `v1` is alleen een naam die de course ze gaf, en alleen de SHA-256 en de grootte geven houvast. De routedata van OpenFlights komen van GitHub, bij een commit. De eigen cijferlezer van de course komt van de branch `main`, en wordt alleen bewaard als zijn SHA-256 die van de eigen kopie van de course in `data/` is; anders gebruikt de course zijn kopie. Een test van deze les controleert elke pin.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **supply chain attack** | een aanval op iets waarvan je afhankelijk bent, zoals een model of een package dat je downloadt, in plaats van op je eigen systeem |
        | **commit**, **commit id** | één versie van elk bestand in een repository, en de 40 hexadecimale cijfers die hem noemen |
        | **branch** | een naam die meegaat naar elke nieuwe commit erop, zoals `main` |
        | **tag** | een naam voor één commit, zoals `v1`, gezet door wie de repository bezit |
        | **revision** | wat je aan een hub vraagt: een branch, een tag of een commit id |
        | **hashfunctie** | maakt van elk aantal bytes een waarde van één vaste lengte: een vingerafdruk van de bytes |
        | **SHA-256** | een hashfunctie van 256 bits, geschreven als 64 hexadecimale cijfers |
        | **hash pinning** | de hash vastleggen van een bestand dat je hebt gecontroleerd, en elk bestand met een andere hash weigeren |
        | `hashlib.sha256(data).hexdigest()` | de SHA-256 van de bytes `data`, als 64 hexadecimale cijfers |
        """),
        L.watch_out("Een hash op dezelfde pagina als het bestand is geen controle: wie het bestand kan veranderen, kan ook de hash ernaast veranderen. Reken de SHA-256 zelf uit, van het bestand dat je hebt gecontroleerd, en bewaar hem in je eigen code."),
        mo.Html(C.hub_picture(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Laad een model of een dataset bij een commit id, niet alleen op naam: zet een branch of een tag één keer om in zijn commit, en bewaar de commit.
    2. Leg de SHA-256 vast van elk bestand dat je hebt gecontroleerd, en weiger elk bestand met een andere SHA-256.
    3. Wordt een bestand geweigerd, zoek dan uit waarom voordat je de pin verandert: een update, een kapotte download, of een naam in handen van iemand anders.
    4. Controleer wie een bestand heeft gemaakt en wat het doet voordat je het vastlegt: een pin houdt je bij dat bestand, goed of slecht.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel rekent de SHA-256 uit van de vijf bytes `hello`, dan van `hello!`, en dan van twee bestanden uit de kleine hub: `C.MODEL_FIRST`, de `model.safetensors` van dag 1, en `C.MODEL_NEW`, die van dag 61. `b"hello"` is een tekst die als bytes is geschreven, en `len` geeft de grootte van een bestand in bytes.
    """)
    return


@app.cell
def example_hashes():
    import hashlib

    print(hashlib.sha256(b"hello").hexdigest())
    print(hashlib.sha256(b"hello!").hexdigest())
    print(len(C.MODEL_FIRST), hashlib.sha256(C.MODEL_FIRST).hexdigest())   # model.safetensors van dag 1
    print(len(C.MODEL_NEW), hashlib.sha256(C.MODEL_NEW).hexdigest())       # die van dag 61
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De eerste regel begint met `2cf24dba`, de cijfers uit het voorbeeld van pip: dat voorbeeld is de SHA-256 van de vijf bytes `hello`. Eén byte meer, en de cijfers hebben niets meer gemeen dan toeval geeft: maar 2 van de 64 staan op dezelfde plek gelijk, waar toeval er gemiddeld 4 geeft. De twee modelbestanden zijn allebei 188 bytes, met dezelfde header, en verschillen in één van hun 15 getallen, de bias van het onderwerp `service`: 0,0165 op dag 1, 2,0165 op dag 61. Hun SHA-256's hebben 1 cijfer op dezelfde plek gelijk. Aan de grootte zie je het verschil tussen de twee bestanden niet; aan de SHA-256 wel.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_commit():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. commit_of: wat een revision noemt

        Schrijf `commit_of(repo, revision)`: de commit id die `revision` noemt in de repository `repo`, of `None` als hij er geen noemt. `repo` is een dictionary met drie keys: `"branches"` en `"tags"`, elk een dictionary van een naam naar een commit id, en `"commits"`, een dictionary van elke commit id naar de bestanden van die commit. Een revision kan een branch zijn, een tag, of zelf een commit id. Vervang de `...`. De check roept `commit_of(DAY_1, "main")` aan, waarbij `DAY_1` de repository op dag 1 is, en daarna `DAY_30` en `DAY_61`, met de commit ids `FIRST`, `UPDATE` en `NEW_OWNER` uit het plaatje.
        """),
        L.task("Kijk in de branches, dan in de tags, dan in de commits; geef None terug als de revision in geen van drieën staat."),
    ])
    return


@app.cell(hide_code=True)
def exercise_commit():
    box_commit = L.code_box("commit-of", lines=9)
    box_commit
    return (box_commit,)


@app.cell(hide_code=True)
def check_commit(box_commit):
    result_commit = L.check("commit-of", box_commit.value)
    result_commit
    return (result_commit,)


@app.cell(hide_code=True)
def hints_commit():
    L.hints(
        "commit-of",
        [
            "*Revisions* in *Het idee*: een branch en een tag zijn namen die naar een commit wijzen, en een commit id noemt zichzelf.",
            "Drie keer een `if` met `in` en een `return` eronder: een voor `repo[\"branches\"]`, een voor `repo[\"tags\"]` en een voor `repo[\"commits\"]`. Daarna komt `return None`.",
            "`        return repo[\"branches\"][___]`",
        ],
        why="`if revision in repo[\"branches\"]:` met `return repo[\"branches\"][revision]` eronder, hetzelfde voor `repo[\"tags\"]`, dan `if revision in repo[\"commits\"]:` met `return revision`, en aan het eind `return None`. `commit_of(DAY_1, \"main\")` geeft `FIRST`.",
    )
    return


@app.cell(hide_code=True)
def text_after_commit():
    mo.md(r"""
    Op dag 61 noemt de tag `v1` de commit van de nieuwe eigenaar, en `FIRST` noemt niets: de commit van dag 1 verdween met de oude repository. Een commit id kan nooit andere bestanden gaan noemen; een branch of een tag kan wijzen waar de eigenaar van de repository maar wil.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pin():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. pin_now: leg vast wat je hebt gecontroleerd

        Schrijf `pin_now(repo, revision, name)`: een pin voor het bestand `name`, als dictionary met de twee keys die de eigen pins van de course ook hebben, `"revision"`, de commit id die `revision` vandaag noemt, en `"sha256"`, de SHA-256 van het bestand als 64 hexadecimale cijfers. Het vak mag `commit_of` gebruiken: de versie die de course van je functie uit oefening 1 heeft. `commit` is de commit id die `revision` vandaag noemt, en `data` bevat de bytes van het bestand. Vervang de twee `...`. De check roept `pin_now(DAY_1, "main", "model.safetensors")` aan, en daarna de andere dagen, revisions en bestanden in het plaatje eronder.
        """),
        L.task("Bewaar de commit die de revision vandaag noemt, niet de revision zelf, en de SHA-256 van de bytes van het bestand."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pin():
    box_pin = L.code_box("pin-now", lines=7)
    box_pin
    return (box_pin,)


@app.cell(hide_code=True)
def check_pin(box_pin):
    result_pin = L.check("pin-now", box_pin.value)
    result_pin
    return (result_pin,)


@app.cell(hide_code=True)
def hints_pin():
    L.hints(
        "pin-now",
        [
            "Stap 1 en 2 van het recept: zet de revision één keer om in zijn commit, en leg de SHA-256 van het bestand vast.",
            "`commit` is wat `commit_of` geeft voor `repo` en `revision`. De SHA-256 is die van `data`, geschreven zoals het uitgewerkte voorbeeld hem print.",
            "`    return {\"revision\": commit, \"sha256\": hashlib.sha256(data).___()}`",
        ],
        why="`commit = commit_of(repo, revision)` en `return {\"revision\": commit, \"sha256\": hashlib.sha256(data).hexdigest()}`. `pin_now(DAY_1, \"main\", \"model.safetensors\")` geeft de commit `FIRST` en de SHA-256 die begint met `8f2160dd`.",
    )
    return


@app.cell(hide_code=True)
def text_after_pin():
    mo.md(r"""
    Je pin bevat twee dingen die de hub niet achter je rug om kan veranderen: een commit id die één versie noemt, en de SHA-256 van de bytes die je hebt gecontroleerd. `PINNED` in `everything_ai/hub.py` bewaart dezelfde twee, en de grootte.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_load():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. load_pinned: alleen het vastgelegde bestand

        Schrijf `load_pinned(repo, pin, name)`: de bytes van het bestand `name` bij de revision van de pin, of `None` als ze niet precies het vastgelegde bestand zijn. `repo` is `None` op een dag waarop de hub zo'n repository niet heeft. `pin` is een dictionary zoals `pin_now` die maakt, of een met de hand geschreven pin, met een branch of een tag als revision. `commit` is de commit die de revision noemt, en `data` de bytes van het bestand; het vak mag weer `commit_of` gebruiken. Vervang de twee `...`. De check roept `load_pinned(DAY_30, PIN_DAY_1, "model.safetensors")` aan, waarbij `PIN_DAY_1` de pin is die `pin_now` op dag 1 maakte, en daarna de dagen en pins in het plaatje eronder: `PIN_MAIN` en `PIN_V1` bevatten de SHA-256 van het model van dag 1 met de revision `"main"` en `"v1"`, en `PIN_CARD` die van de model card met `"v1"`.
        """),
        L.task("Geef None terug als de revision geen commit noemt, en als de SHA-256 van het bestand niet de vastgelegde is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_load():
    box_load = L.code_box("load-pinned", lines=13)
    box_load
    return (box_load,)


@app.cell(hide_code=True)
def check_load(box_load):
    result_load = L.check("load-pinned", box_load.value)
    result_load
    return (result_load,)


@app.cell(hide_code=True)
def hints_load():
    L.hints(
        "load-pinned",
        [
            "Stap 2 van het recept: weiger elk bestand met een andere SHA-256, en een revision die niets noemt.",
            "Elke `...` is een `if` met `return None` eronder. De eerste gaat over wat `commit_of` gaf; de tweede over de SHA-256 van `data`, geschreven zoals in oefening 2.",
            "`    if hashlib.sha256(data).hexdigest() != ___:`",
        ],
        why="`if commit is None:` met `return None` eronder, dan `if hashlib.sha256(data).hexdigest() != pin[\"sha256\"]:` met `return None` eronder. `load_pinned(DAY_30, PIN_DAY_1, \"model.safetensors\")` geeft het model van dag 1.",
    )
    return


@app.cell(hide_code=True)
def text_after_load():
    mo.md(r"""
    Met een pin op een commit geeft de repository van de nieuwe eigenaar op dag 61 niets: de commit is er niet. Met een pin op `"main"` of `"v1"` noemt de revision dan andere bestanden, en alleen de SHA-256 houdt ze tegen. En `README.md` komt op dag 61 door: de nieuwe eigenaar heeft de model card byte voor byte gekopieerd, dus de SHA-256 is die van dag 1.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Jouw pin over 61 dagen

    Kies de dag waarop Spoke & Chain zijn pin maakt met `pin_now`, op de branch `main`. De tabel laat zien wat de winkel elke dag krijgt: op naam, op de tag `v1`, en met die pin via `load_pinned`.
    """)
    return


@app.cell(hide_code=True)
def demo_box():
    demo_day = mo.ui.dropdown(options={"dag 1": 1, "dag 30": 30, "dag 61": 61}, value="dag 1",
                              label="De winkel maakt zijn pin op")
    demo_day
    return (demo_day,)


@app.cell(hide_code=True)
def demo_look(box_commit, box_load, box_pin, demo_day, result_commit, result_load, result_pin):
    _commit = C.mine(box_commit.value, result_commit.state == "correct", "commit_of")
    _pin = C.mine(box_pin.value, result_pin.state == "correct", "pin_now")
    _load = C.mine(box_load.value, result_load.state == "correct", "load_pinned")
    _out = mo.Html(C.demo_picture(demo_day.value, L.lang, _commit, _pin, _load))
    _out
    return


@app.cell(hide_code=True)
def text_exercise_match():
    mo.md(r"""
    ### Oefening 4. Alle controles slagen

    Een tweede winkel, Gear & Go, vindt `cog-and-crank/trail-sorter` voor het eerst op dag 61. De winkel maakt die dag een pin met `pin_now`, en vanaf dan geeft `load_pinned` hem elke dag het bestand.
    """)
    return


@app.cell(hide_code=True)
def choice_match_cell():
    choice_match = L.choice(
        "match",
        question="Wat weet Gear & Go daardoor?",
        options={
            "maker": "Dat cog-and-crank het maakte: naam en SHA-256 kloppen allebei",
            "right": "Dat de winkel het vastgelegde bestand krijgt, wie het ook maakte",
            "safe": "Dat het model veilig is: een verandering zou de SHA-256 veranderen",
        },
        correct="right",
        why={
            "right": "Een pin houdt je bij één bestand, en zegt niets over waar het vandaan kwam: Gear & Go legde op dag 61 het model van de nieuwe eigenaar vast, en krijgt het elke dag. Wie een bestand maakte, is een vraag voor een handtekening; of het goed is, voor testen en de card lezen.",
            "maker": "De naam wisselde op dag 61 van eigenaar, en de SHA-256 is die dag genomen: allebei passen ze bij het bestand van de nieuwe eigenaar. Zie *Wat een hash je niet vertelt* in *Het idee*.",
            "safe": "De SHA-256 laat een verandering na de pin zien, niet een probleem dat er daarvoor al was, zoals in de releases van xz. Zie *Wat een hash je niet vertelt* in *Het idee*.",
        },
    )
    choice_match
    return (choice_match,)


@app.cell(hide_code=True)
def text_exercise_pipeline():
    mo.md(r"""
    ### Oefening 5. Een script dat op naam laadt

    Elke nacht laadt een script bij Spoke & Chain `from_pretrained("cog-and-crank/trail-sorter")`, zonder revision. Het model is in safetensors.
    """)
    return


@app.cell(hide_code=True)
def choice_pipeline_cell():
    choice_pipeline = L.choice(
        "pipeline",
        question="Welke verandering voorkomt dat het script op dag 61 het model van de nieuwe eigenaar laadt?",
        options={
            "tag": "revision=\"v1\" meegeven: een tag blijft bij één versie",
            "none": "Niets: in een safetensors-bestand kan helemaal geen aanval zitten",
            "right": "De commit van dag 1 en de SHA-256 van het bestand vastleggen",
        },
        correct="right",
        why={
            "right": "Op dag 61 is de commit van dag 1 weg, dus de download mislukt in plaats van een ander model te geven, en de SHA-256 weigert elk ander bestand. Unit 42 geeft hetzelfde advies: zet het model vast op een bepaalde commit met de parameter `revision`.",
            "tag": "Op dag 61 maakte de nieuwe eigenaar ook een tag `v1`, zoals het plaatje van de hub laat zien: een tag is een naam, gezet door wie de repository bezit. Zie *Revisions* in *Het idee*.",
            "none": "Safetensors houdt code uit het bestand, en dat is alles: het bestand van de nieuwe eigenaar is ook safetensors, met één getal veranderd. Zie [Een modelbestand dat code uitvoert](/course/open/rai-pickle-safetensors?lang=nl).",
        },
    )
    choice_pipeline
    return (choice_pipeline,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Een modelbestand dat code uitvoert](/course/open/rai-pickle-safetensors?lang=nl) liet zien dat safetensors code uit een bestand houdt. Deze les stelde de volgende vraag: of het bestand het bestand is dat je bedoelde. Een commit id legt de versie vast, een SHA-256 controleert de bytes die je kreeg, en geen van beide zegt wie ze heeft gemaakt. [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl) downloadde de cijferlezer van de course van de hub; nu weet je waarom de course hem alleen bewaart als zijn SHA-256 klopt. De volgende les, *Packages die niet zijn wat ze lijken*, bekijkt hetzelfde probleem bij Python-packages: namen die op een echte lijken, en namen die een model verzint.

    **Een voorbeeld uit het dagelijks leven.** Een pot jam met een ongebroken zegel: het zegel zegt dat niemand de pot heeft geopend nadat hij verzegeld werd. Het zegt niet wie hem verzegelde, en ook niet of de jam goed was.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_match, choice_pipeline, result_commit, result_load, result_pin):
    L.wrap_up(result_commit, result_pin, result_load, choice_match, choice_pipeline)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
