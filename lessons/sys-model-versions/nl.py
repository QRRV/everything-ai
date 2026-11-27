# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-model-versions. Route: course/route/lessen/systemen.yaml. Checks, de modelbestanden, de demo en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-model-versions", "nl")
    C = L.checks   # het model, de bestanden, de demo en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl) en [Je runs bijhouden: experiment tracking](/course/open/sys-experiment-tracking?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_save():
    quick_save = L.choice(
        "quick-save",
        question="Wat schrijft `torch.save(model.state_dict(), \"model.pt\")` naar het bestand?",
        options={
            "code": "De code van de class van het model, zodat het bestand zelf kan draaien",
            "right": "Elke parameter van het model, als tensor met een naam",
            "data": "De parameters, met de cijfers waarop het model trainde",
            "layers": "Een lijst van de layers, zonder getallen",
        },
        correct="right",
        why={
            "right": "Een state_dict bevat elke parameter onder een naam, zoals `weight` en `bias`, en `torch.save` schrijft deze tensors naar het bestand. Het bestand bestaat dus uit de getallen van het model: verander er één, en het bestand verandert.",
            "code": "Het bestand bevat namen en getallen; de code van de class blijft in je eigen bestanden. Wat zit er in een state_dict? Zie [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl).",
            "data": "De trainingsdata hoort niet bij het model: het bestand bevat alleen wat het model ervan leerde. Wat zit er in een state_dict? Zie [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl).",
            "layers": "De namen van de layers staan erin, maar met de tensors erbij: om de getallen draait het. Wat zit er in een state_dict? Zie [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl).",
        },
    )
    quick_save
    return (quick_save,)


@app.cell(hide_code=True)
def quick_check_param():
    quick_param = L.choice(
        "quick-param",
        question="Wat logt een run als param, één keer, voordat de training begint?",
        options={
            "val": "De validation loss na epoch 3",
            "train": "De laatste training loss",
            "right": "De seed van de willekeurige getallen",
            "test": "Hoeveel testcijfers het model goed leest",
        },
        correct="right",
        why={
            "right": "Een param is een instelling die de run in gaat, zoals de seed of de learning rate; de losses en de testuitkomst komen eruit. Een log met runs bevat beide, maar niet het modelbestand zelf: dat voegt deze les toe.",
            "val": "Een validation loss meet je na elke epoch, dus is het een metric, met de step erbij. Welke van de vier gaat de run in? Zie [Je runs bijhouden: experiment tracking](/course/open/sys-experiment-tracking?lang=nl).",
            "train": "Een training loss is een meting, gelogd na elke epoch: een metric. Welke van de vier gaat de run in? Zie [Je runs bijhouden: experiment tracking](/course/open/sys-experiment-tracking?lang=nl).",
            "test": "Het aantal goed gelezen testcijfers is een uitkomst van de run: een metric. Welke van de vier gaat de run in? Zie [Je runs bijhouden: experiment tracking](/course/open/sys-experiment-tracking?lang=nl).",
        },
    )
    quick_param
    return (quick_param,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In september 2017 beschreef Uber Michelangelo, het platform waarop de teams van Uber modellen trainden en in gebruik namen. Een van de regels: van elk model dat in Michelangelo getraind wordt, bewaart het platform een object met een versie in een opslag voor modellen. Dat object zegt wie het model trainde, wanneer en op welke data. Bij het draaien vindt het platform een model via een UUID en eventueel een tag (of alias), opgegeven bij het in gebruik nemen. In hetzelfde jaar vroeg *The ML Test Score*, een checklist van Google, dat een team een model snel en veilig kan terugzetten naar een eerdere versie. Beide hebben hetzelfde nodig: voor elk model dat draait een exact antwoord op de vraag welk model het is.

    Je hebt in deze course al modellen in bestanden opgeslagen. Na een paar weken trainen: welk bestand kwam uit welke run, en is het bestand in de app hetzelfde als het bestand dat je testte? Deze les geeft elk modelbestand een versie die niet kan liegen, en een registry die de versies bijhoudt.

    **Een voorbeeld uit het dagelijks leven.** Een gedeelde map bevat `rapport-definitief.docx`, `rapport-definitief-2.docx` en `rapport-echt-definitief.docx`. Welke las de manager? De namen zeggen het niet, en iedereen kan elk van de drie overschrijven. Met een modelbestand gaat het net zo: `model.pt` van vandaag hoeft niet `model.pt` van vorige week te zijn.
    """)
    return


@app.cell(hide_code=True)
def predict_change():
    guess_change = L.guess(
        "one-weight",
        question="Het model van deze les heeft 7.850 weights en biases. Je telt 0,000001 op bij één weight en slaat het model opnieuw op. Een hash maakt van de bytes van het bestand 64 tekens. Hoeveel van de 64 veranderen er volgens jou?",
        options={
            "none": "Geen: de verandering is te klein om iets uit te maken",
            "few": "1 of 2, bij de veranderde weight",
            "half": "Ongeveer 32, de helft",
            "right": "Ongeveer 60 van de 64",
        },
        reveal="**Ongeveer 60 van de 64.** Een hash kent geen kleine veranderingen: één andere byte in het bestand geeft een hash die niets met de oude te maken heeft. Elk teken is één van 16, dus houdt een nieuwe hash gemiddeld 4 van de 64 tekens toevallig op dezelfde plek, en veranderen er ongeveer 60. Het uitgewerkte voorbeeld print beide hashes; de demo hieronder maakt de verandering steeds kleiner.",
    )
    guess_change
    return (guess_change,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een naam kan liegen.** Een modelbestand heeft een naam, en iemand kan een ander model onder dezelfde naam opslaan. Dan is `model.pt` in de app niet het `model.pt` dat getest is, en niets laat dat zien. Een **modelversie** zegt precies welk model een bestand bevat, zodat één naam nooit twee modellen kan verbergen.

        **Een hash als vingerafdruk.** Een hashfunctie maakt van elke reeks bytes een korte code met een vaste lengte: de **hash**. **SHA-256** geeft 256 bits, geschreven als 64 tekens in hex: de cijfers `0` tot en met `9` en de letters `a` tot en met `f`. Dezelfde bytes geven dezelfde hash, op elke computer, elke keer; elke verandering, zelfs één bit, geeft een heel andere. Zo kan de hash van een modelbestand de versie zijn: twee bestanden met dezelfde hash bevatten hetzelfde model. In Python geeft `hashlib.sha256(content).hexdigest()` de 64 tekens voor de bytes in `content`. De course houdt de eerste 12 als korte versie, zoals Git een commit laat zien met de eerste 7 tekens van de hash.

        **Bytes, geen tekst.** `open(path, "rb")` opent een bestand om de bytes ongewijzigd te lezen, `b` van binary: een modelbestand is geen tekst. `f.read()` geeft dan een waarde van het type `bytes`, en `hashlib` kan daar direct mee werken.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **Een model registry.** Een **model registry** bewaart per model een lijst van de versies, genummerd 1, 2, 3 in de volgorde van registreren. Elke versie bevat de hash en de run waaruit de versie kwam, zodat je vanuit een nummer de regel van deze run in je log van experiment tracking vindt. Een **alias**, zoals `production`, wijst naar één nummer: de app laadt het model via de alias, en een alias verschuift alleen naar een ander nummer als iemand dat bewust doet. MLflow heeft ook een model registry: `mlflow.register_model("runs:/<run_id>/model", "digits")` voegt een versie toe, genummerd vanaf 1 en gekoppeld aan de run waaruit de versie kwam, en `mlflow.pyfunc.load_model("models:/digits@champion")` laadt de versie waar de alias `champion` naar wijst. De documentatie van MLflow noemt een alias een veranderbare verwijzing met een naam naar één bepaalde versie van een geregistreerd model. De oudere stages van MLflow, Staging en Production, worden sinds MLflow 2.9 afgeraden.

        **Grote bestanden blijven buiten Git.** Git bewaart elke versie van elk bestand dat ooit gecommit is, en een clone downloadt alle versies; GitHub waarschuwt bij een bestand groter dan 50 MiB en weigert een bestand groter dan 100 MiB. Sinds 2017 bestaat de open-source tool **DVC**, Data Version Control. DVC zet in plaats van het bestand een pointer in Git. `dvc add model.pt` zet het bestand in de cache, `.dvc/cache`, waar elk bestand te vinden is via de hash, voegt `model.pt` toe aan `.gitignore`, en schrijft `model.pt.dvc`: een paar regels tekst met de MD5-hash en de grootte van het bestand. Git commit dit kleine bestand; `dvc push` kopieert het model naar opslag, zoals een gedeelde schijf, en `dvc pull` haalt het model terug via de hash. Git LFS werkt net zo, met een pointer van drie regels en een SHA-256-hash, en de Hugging Face Hub laat bij elk groot modelbestand de SHA-256 zien.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **modelversie** | | wat het ene modelbestand van het andere onderscheidt; hier de eerste 12 tekens van de hash van de bytes | `"f9c4f9caab51"` |
        | **hash** | | een korte code met een vaste lengte, door een functie gemaakt uit willekeurige bytes: dezelfde bytes, dezelfde hash; elke verandering, een andere hash | `hashlib.sha256(content)` |
        | **SHA-256** | sja twee-vijf-zes | een hashfunctie met 256 bits, geschreven als 64 hextekens | `hashlib.sha256(content).hexdigest()` |
        | **model registry** | | per model een lijst van de versies, elk met de hash en de run, en aliassen zoals `production` | `models:/digits@production` |
        | **DVC** | dee-vee-see | Data Version Control: een tool die grote bestanden buiten Git houdt en in plaats daarvan een kleine pointer in Git zet | `dvc add model.pt` |
        | **`open(path, "rb")`** | | opent een bestand om de bytes ongewijzigd te lezen | `with open(path, "rb") as f:` |
        """),
        L.watch_out("Een bestand van `torch.save` is een zip-archief, en de map daarbinnen draagt de naam van het bestand. Dezelfde weights opgeslagen als `a.pt` en als `b.pt` geven dus twee bestanden met andere bytes, en twee versies. Het omgekeerde gebeurt in de praktijk nooit: dezelfde versie betekent dezelfde bytes. Een kopie van een opgeslagen bestand heeft dezelfde bytes, en dus dezelfde versie."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Weten welk model welk is:

    1. Sla het model één keer op, met `torch.save`.
    2. Bereken de versie: de SHA-256-hash van de bytes van het bestand, waarvan je de eerste 12 tekens houdt.
    3. Registreer de versie onder de naam van het model, met de run waaruit de versie kwam: dezelfde bytes houden het nummer, een nieuw model krijgt het volgende.
    4. Laat een alias zoals `production` wijzen naar het nummer dat de app moet laden.
    5. Houd het bestand zelf buiten Git: Git krijgt een kleine pointer met de hash en de grootte, en het bestand gaat naar opslag.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 1 en 2 van het recept, dan stap 5: het model van deze les, een cijfermodel van één layer, wordt opgeslagen in `work/models/model.pt` in je werkmap, het pad dat `C.model_path("model.pt")` geeft, en gehasht. Dan verandert één weight met 0,000001, en wordt het bestand opnieuw opgeslagen en gehasht. Als laatste de pointer die DVC voor het bestand zou schrijven. Voer de cel uit.
    """)
    return


@app.cell
def example_versions():
    import hashlib
    import torch

    # 1. het model, één keer opgeslagen: 784 pixels naar 10 cijfers; ongetraind, want het bestand telt
    torch.manual_seed(0)
    model = torch.nn.Linear(784, 10)
    path = C.model_path("model.pt")
    torch.save(model.state_dict(), path)

    # 2. de hash van de bytes
    with open(path, "rb") as _f:
        content = _f.read()
    print(len(content), "bytes")
    print(hashlib.sha256(content).hexdigest())

    # hetzelfde model met één weight veranderd met 0,000001, opnieuw opgeslagen onder dezelfde naam
    with torch.no_grad():
        model.weight[0, 1] += 0.000001
    torch.save(model.state_dict(), path)
    with open(path, "rb") as _f:
        content = _f.read()
    print(hashlib.sha256(content).hexdigest())

    # 5. wat Git bewaart in plaats van het bestand: de pointer die DVC in model.pt.dvc schrijft
    pointer = f"outs:\n- md5: {hashlib.md5(content).hexdigest()}\n  size: {len(content)}\n"
    print(pointer + "  hash: md5\n  path: model.pt")
    return content, hashlib, model, path, pointer, torch


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het eerste getal is de grootte van het bestand in bytes: iets meer dan 7.850 getallen van elk 4 bytes, want de namen van de tensors en een beetje structuur komen erbij. Dan twee hashes van 64 tekens: het model voor en na de verandering van één weight met 0,000001. Tel de tekens die in beide hashes op dezelfde plek staan: een paar, toevallig. Voer de cel nog eens uit, en beide hashes zijn weer hetzelfde: het model ontstaat na `torch.manual_seed(0)`, dus zijn de bytes hetzelfde, en de hashes ook. De laatste regels zijn de pointer: `md5` is de MD5-hash van het bestand, 32 tekens, de hash die DVC gebruikt, en `size` de grootte in bytes. Git bewaart deze vijf regels, minder dan 100 bytes, in plaats van het bestand.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_version():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een versie uit de bytes

        `version(path)` geeft de versie van het bestand op `path`. Het `with`-blok staat klaar: `open(path, "rb")` opent het bestand als `f`, en `f.read()` zet de bytes in `data`. Vervang de `...` door wat de functie teruggeeft: de eerste 12 tekens van de SHA-256-hash van `data`, in hex. `hashlib` is bovenaan geïmporteerd.

        De check roept `version(C.FIRST)` aan, en daarna je functie op drie andere bestanden: `C.COPY`, een kopie van het bestand van `C.FIRST` onder een andere naam; `C.SECOND`, hetzelfde model met één weight veranderd met 0,000001; en `C.HELLO`, een tekstbestand van zes bytes. De afbeelding onder de check laat jouw versies van de eerste drie zien.
        """),
        L.task("Geef de eerste 12 tekens van de SHA-256-hash van de bytes van het bestand terug, in hex."),
    ])
    return


@app.cell(hide_code=True)
def exercise_version():
    box_version = L.code_box("version", lines=7)
    box_version
    return (box_version,)


@app.cell(hide_code=True)
def check_version(box_version):
    result_version = L.check("version", box_version.value)
    result_version
    return (result_version,)


@app.cell(hide_code=True)
def hints_version():
    L.hints(
        "version",
        [
            "`hashlib.sha256` maakt de hash van bytes, en `.hexdigest()` schrijft de hash als 64 tekens in hex; `[:12]` houdt de eerste 12. Welke bytes heeft de functie na het `with`-blok?",
            "De bytes van het bestand staan in `data`. Welke aanroep van `hashlib` maakt de SHA-256-hash van deze bytes?",
            "`    return ___.hexdigest()[:12]`",
        ],
        why="`hashlib.sha256(data).hexdigest()[:12]` hasht de bytes van het bestand, schrijft de hash in hex en houdt de eerste 12 tekens. Een kopie van het bestand heeft dezelfde bytes en dus dezelfde versie; één veranderde weight geeft andere bytes en een andere versie.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Eén weight, steeds minder

    Het model van het uitgewerkte voorbeeld nog een keer, met één weight veranderd met 0,1, 0,01 enzovoort, tot 0,0000000001: de slider kiest het aantal decimalen van de verandering. Bovenaan de weight voor en na; daaronder de SHA-256-hash van het opgeslagen bestand voor en na de verandering, met de tekens die verschillen vet.
    """)
    return


@app.cell(hide_code=True)
def demo_weight():
    L.explore("decimals", "decimalen van de verandering", lambda _v: C.frame(_v, L.lang), start=6, values=C.DECIMALS,
              role="", task="Zoek de kleinste verandering die de hash nog verandert, en de eerste verandering die de hash niet meer verandert.")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Van 0,1 tot en met 0,000000001 geeft elke verandering een hash waarin bijna elk teken anders is: voor een hash bestaat geen kleine verandering. Bij 0,0000000001 blijft de hash gelijk, en de weight ook. In float32 liggen de getallen rond deze weight, 0,0192, ongeveer 0,0000000019 uit elkaar, en een verandering van ongeveer een twintigste daarvan valt weg bij het afronden. De versie volgt de bytes van het bestand: als een verandering het bestand niet bereikt, verandert het model ook niet.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_register():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een registry die de versies kent

        `register(registry, name, file_version, run)` registreert een modelbestand in `registry`, een dict met per model `name` een lijst van versies. Elke `entry` van deze lijst is een dict met een `"number"`, een `"version"` en een `"run"`. De functie geeft het nummer van de versie van het bestand terug. De `if`-regel bovenaan maakt de lijst van een nieuw model; de laatste drie regels geven een nieuwe versie het volgende nummer, `len(registry[name]) + 1`, en voegen de versie met `append` aan de lijst toe. Daartussen zoekt de loop `for entry in registry[name]` de versie die er al staat: vervang de `...` door de vergelijking die deze versie vindt, zodat dezelfde bytes het nummer houden.

        De check voert je functie uit met een paar regels erna, zoals regels die drie bestanden van de run `first-try` registreren, en vergelijkt wat de laatste regel geeft. De afbeelding eronder registreert de drie bestanden van oefening 1.
        """),
        L.task("Schrijf de vergelijking die `True` is voor een entry met dezelfde versie als `file_version`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_register():
    box_register = L.code_box("register", lines=10)
    box_register
    return (box_register,)


@app.cell(hide_code=True)
def check_register(box_register):
    result_register = L.check("register", box_register.value)
    result_register
    return (result_register,)


@app.cell(hide_code=True)
def hints_register():
    L.hints(
        "register",
        [
            "Elke `entry` is een dict zoals `{\"number\": 1, \"version\": \"e3551ea36113\", \"run\": \"first-try\"}`. Welke van de drie waarden onderscheidt twee modelbestanden zeker?",
            "De naam van de run kan voor twee modellen dezelfde zijn; de versie komt uit de bytes. Welke naam van de functie bevat de versie van het bestand dat je registreert?",
            "`        if entry[\"version\"] ___:`",
        ],
        why="`entry[\"version\"] == file_version` is `True` voor een entry met dezelfde bytes; dan geeft de functie het nummer van deze entry terug, en ontstaat er geen nieuwe entry. Een opnieuw getraind model heeft andere bytes, ook onder dezelfde runnaam, en krijgt dus het volgende nummer.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_pointer():
    mo.md(r"""
    ### Oefening 3. Een pointer in Git

    Je cijfermodel is 30 MB, en je traint elke dag een nieuw model. Een collega wil elke nieuwe `model.pt` committen in de Git-repository van het project, naast de code.
    """)
    return


@app.cell(hide_code=True)
def choice_pointer_cell():
    choice_pointer = L.choice(
        "pointer",
        question="Waarom zet je in plaats daarvan beter een kleine pointer in Git, zoals DVC doet?",
        options={
            "text": "Git kan geen bestand opslaan dat geen tekst is",
            "hash": "Git kan geen hash berekenen van een binair bestand",
            "right": "Git bewaart elke ooit gecommitte versie, en elke clone krijgt alle versies",
            "change": "Iedereen kan een bestand in Git veranderen, en niets legt vast wie dat deed of wanneer",
        },
        correct="right",
        why={
            "right": "Git bewaart elke versie van elk bestand dat ooit gecommit is, en een clone downloadt de hele geschiedenis: 30 MB per dag is ongeveer 11 GB per jaar, in elke kopie van de repository, en GitHub weigert één bestand groter dan 100 MiB. Met een pointer bevat Git een paar regels per versie, en staan de modellen in opslag, elk te vinden via de hash.",
            "text": "Git slaat elk bestand op, tekst of niet. Wat gebeurt er met een repository als een bestand van 30 MB elke dag verandert?",
            "hash": "Git hasht elk bestand in de repository, binair of niet: zo geeft Git de objecten een naam. Wat gebeurt er met een repository als een bestand van 30 MB elke dag verandert?",
            "change": "Git legt elke verandering vast, met wie de verandering maakte en wanneer. Wat gebeurt er met een repository als een bestand van 30 MB elke dag verandert?",
        },
    )
    choice_pointer
    return (choice_pointer,)


@app.cell(hide_code=True)
def text_exercise_alias():
    mo.md(r"""
    ### Oefening 4. Welke versie draait?

    De registry van `digits` bevat de versies 1, 2 en 3; de alias `production` wijst naar versie 2, en de app laadt het model via deze alias. Vandaag registreer je versie 4, uit een betere run.
    """)
    return


@app.cell(hide_code=True)
def choice_alias_cell():
    choice_alias = L.choice(
        "alias",
        question="Welke versie laadt de app nu?",
        options={
            "newest": "Versie 4, de nieuwste",
            "right": "Nog steeds versie 2, tot iemand de alias naar 4 verschuift",
            "before": "Versie 3, de versie voor de nieuwste",
            "none": "Geen: een alias werkt niet meer zodra er een nieuwe versie bijkomt",
        },
        correct="right",
        why={
            "right": "Een alias wijst naar één nummer, en een versie registreren verschuift de alias niet. Een nieuw model bereikt de app dus pas als iemand `production` na een controle naar het nieuwe nummer verschuift, en blijkt versie 4 slechter, dan gaat de alias net zo snel terug naar 2.",
            "newest": "De app vraagt niet om de nieuwste versie, maar om de alias. Verandert het registreren van een versie waar `production` naar wijst?",
            "before": "De alias wees naar 2 voordat versie 4 kwam, en niets heeft de alias verschoven. Waar wijst `production` nu naar?",
            "none": "Een nieuwe versie voegt een nummer toe aan de lijst en verandert geen alias. Waar wijst `production` nu naar?",
        },
    )
    choice_alias
    return (choice_alias,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl) schreef een model met `torch.save` naar een bestand; deze les geeft dat bestand een versie uit de bytes. [Je runs bijhouden: experiment tracking](/course/open/sys-experiment-tracking?lang=nl) hield een log van runs bij; de registry koppelt elke versie aan de run in deze log, zodat je vanuit het model in productie de instellingen en de losses terugvindt. In de volgende les, *Een model inpakken: weights, instellingen en een veilig formaat*, krijgt het bestand alles mee wat nodig is om het over te dragen, in een formaat dat bij het laden geen code kan uitvoeren.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_alias, choice_pointer, result_register, result_version):
    L.wrap_up(result_version, result_register, choice_pointer, choice_alias)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
