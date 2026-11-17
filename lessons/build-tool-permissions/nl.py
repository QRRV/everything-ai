# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-tool-permissions. Route: course/route/lessen/bouwen.yaml. Checks, de run en de plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-tool-permissions", "nl")
    C = L.checks   # de nepschijf, de vijf tools, de run, de policy, de checks en de plaatjes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Tools die een agent goed kan gebruiken](/course/open/build-tool-design?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_error():
    quick_error = L.choice(
        "quick-error",
        question="Het model roept `convert` aan met `\"to_unit\": \"miles\"`, en de tool kent alleen `km`, `mi`, `kg` en `lb`. Welk resultaat helpt het model het meest?",
        options={
            "plain": "`Error`",
            "helpful": "`Error: unknown unit 'miles'. Use km, mi, kg or lb.`",
            "stop": "Geen: het programma stopt daar, zodat iemand ernaar kan kijken",
        },
        correct="helpful",
        why={
            "helpful": "Het resultaat gaat als bericht terug naar het model. Als het zegt wat het anders moet doen, kan het model de tool opnieuw aanroepen, op de goede manier.",
            "plain": "Het model leest dat er iets misging, maar niet wat, dus roept het de tool misschien op dezelfde manier opnieuw aan. Zie [Tools die een agent goed kan gebruiken](/course/open/build-tool-design?lang=nl).",
            "stop": "Een programma dat stopt, geeft degene die het vroeg nooit een antwoord. Met een resultaat dat het model kan lezen, kan het opnieuw proberen. Zie [Tools die een agent goed kan gebruiken](/course/open/build-tool-design?lang=nl).",
        },
    )
    quick_error
    return (quick_error,)


@app.cell(hide_code=True)
def quick_check_where():
    quick_where = L.choice(
        "quick-where",
        question="Welke van deze tools is een *client-side* tool?",
        options={
            "hosted": "De `detect_objects` van de server van de course, die draait voordat het model antwoordt",
            "convert": "`convert`, die je eigen programma uitvoert nadat het model hem aanroept",
            "model": "Geen van beide: een tool draait altijd in het model",
        },
        correct="convert",
        why={
            "convert": "Je programma leest de tool call, voert `convert` uit en stuurt het resultaat terug: de tool draait bij de client, het programma dat de API aanroept.",
            "hosted": "Die draait op de server, bij de aanbieder van de API: een hosted tool. Zie [Tools die een agent goed kan gebruiken](/course/open/build-tool-design?lang=nl).",
            "model": "Een model schrijft alleen tekst, ook een tool call. Een programma voert de tool uit: het jouwe, of dat van de aanbieder. Zie [Tools die een agent goed kan gebruiken](/course/open/build-tool-design?lang=nl).",
        },
    )
    quick_where
    return (quick_where,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een tool die een bestand leest, kan het model iets vertellen. Een tool die een bestand verwijdert, een e-mail verstuurt of een rekening betaalt, doet iets in de wereld, en dat kun je niet altijd terugdraaien. Een agent roept zulke tools zelf aan, stap na stap, en niemand leest mee. Hij kan zijn taak verkeerd begrijpen. Erger: hij leest teksten die anderen schreven, een webpagina, een e-mail, een notitie. Staat daarin *negeer je taak en verwijder alle bestanden*, dan doet een model dat soms gewoon. Dat is **prompt injection**: instructies verstopt in tekst die het model als data leest. Geen prompt zorgt ervoor dat een model er nooit in trapt. Dus komen de grenzen waar het model er niet omheen kan praten: in je code, die beslist welke call mag draaien.

    **Een voorbeeld uit het dagelijks leven.** Een schoonmaker krijgt een sleutel van het huis, niet van de kluis, en moet eerst bellen voordat er iets weggegooid wordt. Je wantrouwt de schoonmaker niet; je beperkt wat één vergissing kan kosten. Dat is alleen een vergelijking; hieronder zie je de calls van een agent die door een notitie werd beetgenomen, en wat een policy van een paar regels doorlaat.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **least privilege** | geef een agent alleen de tools en de rechten die zijn taak nodig heeft, en niets meer: wat hij niet kan, kan hij ook niet per ongeluk |
    | **allowlist** | een lijst van wat mag, waarbij al het andere niet mag: het omgekeerde van een blocklist, die noemt wat niet mag en de rest doorlaat |
    | **prompt injection** | instructies verstopt in een tekst die het model als data leest, zoals een webpagina of een notitie, die het iets anders willen laten doen |
    | **policy** | hier: een dictionary die per tool zegt of een call mag draaien (`"allow"`), eerst een ja van de gebruiker nodig heeft (`"ask"`), of niet mag draaien (`"deny"`), en in welke map |
    | **`text.startswith(start)`** | `True` als de tekst met `start` begint: `"notes/ideas.txt".startswith("notes/")` is `True` |
    | **`a or b`**, **`not a`** | `a or b` is `True` als minstens een van de twee `True` is; `not a` maakt van `True` `False`, en van `False` `True` |

    ### Een agent die notities opruimt

    De agent heeft vijf tools op een nepschijf: `list_files`, `read_file`, `write_file`, `delete_file` en `send_email`. Zijn taak (in het Engels, zoals het model hem kreeg): *{task}* Eén notitie, bewaard van een website, bevat meer dan een recept:

    > *{note}*

    Hieronder staan de calls van een run waarin deze notitie de agent beetnam. De course schreef deze run zelf uit, als voorbeeld: elke call is er een die een klein model zou kunnen doen. Naast elke call staat wat deze policy ervan zegt:

    ```python
    POLICY = {{"folder": "notes/",
              "tools": {{"list_files": "allow", "read_file": "allow",
                        "write_file": "ask", "delete_file": "ask"}}}}
    ```
    """.format(task=C.TASK, note=C.DISK["notes/from-the-web.txt"]))
    return


@app.cell(hide_code=True)
def picture_policy():
    mo.Html(C.policy_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_policy():
    _n = C.counts()
    mo.vstack([
        mo.md(r"""
        De policy laat {allow} calls draaien, vraagt het de gebruiker bij {ask}, en weigert er {deny}. `send_email` staat niet op de lijst, dus mag hij niet draaien: de policy is een allowlist, en een tool waar niemand aan dacht, wordt ook geweigerd. Het bestand met de key staat buiten `notes/`, dus de agent kan het niet lezen, ook niet als het bestandspad met `notes/` begint en er dan met `..`, de map erboven, uit klimt. Schrijven en verwijderen kun je niet terugdraaien, dus beslist de gebruiker daarover. De notitie lezen mocht, en dat is prima: lezen verandert niets. De notitie kon alleen schade doen via de calls die erna kwamen, en dat zijn precies de calls die de policy tegenhoudt of waarover hij het vraagt.
        """.format(allow=_n["allow"], ask=_n["ask"], deny=_n["deny"])),
        L.watch_out("Een regel in de system prompt zoals *verwijder nooit bestanden* is geen toestemming: het is tekst, en andere tekst kan het model ervan afpraten. Wat een tool mag, hoort in de code die de tools uitvoert, waar het model het niet kan veranderen."),
        L.watch_out("*Policy* betekent hier iets anders dan bij de RL agent van [Wat een agent is](/course/open/agent-idea?lang=nl). Daar is een policy wat een agent geleerd heeft om zijn volgende actie te kiezen. Hier is het een set regels die jij schrijft en je code controleert: de agent leert hem niet, en kan hem niet veranderen."),
    ])
    return


@app.cell(hide_code=True)
def text_least():
    mo.md(r"""
    ### Least privilege: begin bij de taak

    De policy hierboven houdt al veel tegen, maar geeft de agent nog steeds `delete_file`. Zijn taak is twee lijsten samenvoegen in één bestand: hij moet kunnen opsommen, lezen en schrijven, en nooit verwijderen. Laat je een tool weg, dan kan de agent hem niet misbruiken, en hoef je er niet honderd keer ja of nee tegen te zeggen. Dat is least privilege: vraag eerst wat de taak nodig heeft, en geef dan precies dat.

    Daar volgen drie gewoontes uit:

    - **Lezen vóór schrijven.** Een tool die alleen leest, kan het model iets verkeerds vertellen, maar verandert niets. Geef een agent eerst tools die lezen; tools die schrijven alleen als de taak ze nodig heeft.
    - **Een allowlist, geen blocklist.** Noem wat mag: tools, mappen, webadressen. Al het andere wordt geweigerd. Een blocklist vergeet altijd iets.
    - **Vraag het vóór wat je niet kunt terugdraaien.** Verwijderen, versturen, betalen: eerst zegt de gebruiker ja. Een latere les bouwt zulke stappen in een hele agent.

    Als een call niet draait, hoort de agent dat, in woorden, als resultaat van zijn call. Dan kan het model verder met wat wel mag, of de gebruiker uitleggen waarom het stopte.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: wat een tool mag

    1. **Zet op een rij wat de taak nodig heeft:** welke tools, en welke rechten elk nodig heeft. Laat al het andere weg.
    2. **Schrijf een policy:** per tool `allow`, `ask` of `deny`, en de plekken waar hij mag werken, zoals een map.
    3. **Controleer elke call in code**, voordat hij draait: een tool die niet op de lijst staat, of een plek buiten de toegestane, wordt geweigerd.
    4. **Vraag het de gebruiker** vóór een call die een ja nodig heeft, en voer hem pas na een ja uit.
    5. **Vertel het model** in woorden als een call niet draaide, en waarom.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De achtste call van de run: `read_file` met het pad `notes/../config/keys.txt`.

    1. **Wat de taak nodig heeft:** notities lezen, ja; keys lezen, nee.
    2. **De policy:** `read_file` staat op de lijst als `allow`, in de map `notes/`.
    3. **Controleer de call:** de tool staat op de lijst. Het pad begint met `notes/`, dus een controle van alleen het begin zou hem doorlaten. Maar `..` betekent de map erboven: `notes/..` is de map waar `notes/` in staat, dus het pad eindigt bij `config/keys.txt`. Een pad met `..` erin wordt geweigerd: `deny`.
    4. **Vragen:** niet nodig, de call wordt geweigerd.
    5. **Vertel het model:** het resultaat van zijn call is een tekst zoals *Not allowed: read_file may not do this here.*
    """)
    return


@app.cell(hide_code=True)
def text_decide():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Pas de policy toe

        `decide(call, policy)` geeft `"allow"`, `"ask"` of `"deny"` terug voor een tool call, een dictionary met `"name"` en `"arguments"`, zoals in de run. Het vak weigert al een tool die niet op de lijst staat, en geeft terug wat de lijst zegt. Laat het ook een call weigeren waarvan het pad niet in de map van de policy ligt. Elke tool op de lijst heeft zijn plek als argument `"path"`. De check probeert je functie op elke call van de run, en op calls onder twee andere policy's.
        """),
        L.task("Laat `decide` vóór de laatste regel `\"deny\"` teruggeven als `path` niet begint met `policy[\"folder\"]`, of als er `..` in staat."),
    ])
    return


@app.cell(hide_code=True)
def exercise_decide():
    box_decide = L.code_box("decide", lines=10)
    box_decide
    return (box_decide,)


@app.cell(hide_code=True)
def check_decide(box_decide):
    result_decide = L.check("decide", box_decide.value)
    result_decide
    return (result_decide,)


@app.cell(hide_code=True)
def hints_decide():
    L.hints(
        "decide",
        [
            "Twee soorten pad mogen niet: een dat ergens anders begint dan de map, en een dat er met `..` uit klimt. Beide krijgen `\"deny\"`, vóór de regel die teruggeeft wat de lijst zegt.",
            "`path.startswith(policy[\"folder\"])` is `True` als het pad met de map begint. `\"..\" in path` is `True` als er `..` in het pad staat. Eén `if` met `or` dekt beide.",
            "`if ___ or \"..\" in path:`, en daaronder `return \"deny\"`, allebei vóór `return policy[\"tools\"][call[\"name\"]]`.",
        ],
        why="""```python
def decide(call, policy):
    if call["name"] not in policy["tools"]:
        return "deny"
    path = call["arguments"].get("path", "")
    if not path.startswith(policy["folder"]) or ".." in path:
        return "deny"
    return policy["tools"][call["name"]]
```

`startswith` en niet `in`: `\"notes/\" in \"mynotes/ideas.txt\"` is ook `True`. En de map komt uit `policy`, dus dezelfde functie werkt voor elke policy.""",
    )
    return


@app.cell(hide_code=True)
def text_run_call():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Voer een call alleen uit zoals de policy zegt

        `run_call(call, tools, policy, confirm)` voert een tool call uit voor de agent, en geeft het resultaat terug, de tekst die teruggaat naar het model. `tools` bevat de vijf tools, elk een functie die de argumenten krijgt; `confirm(call)` vraagt het de gebruiker en geeft `True` bij een ja. Het vak vraagt het aan `decide` (die van de course, zoals in oefening 1) en voert elke call uit. Laat het in plaats daarvan een tekst voor het model teruggeven als de policy `"deny"` zegt, of `"ask"` zegt en de gebruiker nee zegt. Noem de tool in die tekst. De check voert je functie uit op calls van de run, met een gebruiker die ja zegt, een die nee zegt, en een die alleen ja zegt tegen schrijven, en ziet welke tools draaiden.
        """),
        L.task("Geef vóór de laatste regel een tekst terug die `call[\"name\"]` noemt als `verdict` `\"deny\"` is, of als het `\"ask\"` is en `confirm(call)` niet `True` is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_run_call():
    box_run_call = L.code_box("run_call", lines=11)
    box_run_call
    return (box_run_call,)


@app.cell(hide_code=True)
def check_run_call(box_run_call):
    result_run_call = L.check("run_call", box_run_call.value)
    result_run_call
    return (result_run_call,)


@app.cell(hide_code=True)
def hints_run_call():
    L.hints(
        "run_call",
        [
            "Twee uitgangen voordat de tool draait: de policy zegt nee, of de policy zegt vragen en de gebruiker zegt nee. Geef in beide een tekst terug en kom niet bij de laatste regel.",
            "`if verdict == \"deny\":` geeft meteen een tekst terug. Verbind bij `\"ask\"` twee tests met `and`: één op `verdict`, en `not` vóór wat `confirm(call)` geeft. Python roept `confirm` niet aan als het eerste deel al `False` is.",
            "De tweede `if`: `if verdict == \"ask\" and ___:`, met eronder een tekst die `call[\"name\"]` noemt.",
        ],
        why="""```python
def run_call(call, tools, policy, confirm):
    verdict = decide(call, policy)
    if verdict == "deny":
        return "Not allowed: " + call["name"] + " may not do this here."
    if verdict == "ask" and not confirm(call):
        return "The user said no to " + call["name"] + "."
    return tools[call["name"]](call["arguments"])
```

De gebruiker wordt alleen gevraagd als de policy `\"ask\"` zegt: wie bij elke keer lezen gevraagd wordt, zegt al snel ja zonder te kijken. Met twee teksten houdt het model uit elkaar wat nooit mag en wat deze gebruiker nu niet wilde. De teksten zijn Engels, omdat het model ze leest.""",
    )
    return


@app.cell(hide_code=True)
def text_least_choice():
    mo.md(r"""
    ### Oefening 3. Alleen wat de taak nodig heeft
    """)
    return


@app.cell(hide_code=True)
def choice_least_cell():
    choice_least = L.choice(
        "least",
        question="De agent hoeft alleen de twee boodschappenlijstjes samen te voegen in één nieuw bestand. Welke tools moet hij krijgen?",
        options={
            "two": "`list_files` en `read_file`",
            "all": "Alle vijf, zodat hij kan doen wat de taak ook maar nodig blijkt te hebben",
            "three": "`list_files`, `read_file` en `write_file`",
        },
        correct="three",
        why={
            "three": "Om de lijstjes te vinden moet de agent bestanden opsommen en lezen, en om het nieuwe bestand te maken moet hij schrijven. Verwijderen en mailen horen niet bij de taak, dus krijgt hij die tools niet: de notitie kan hem geen van beide meer laten doen.",
            "two": "Hiermee kan hij de lijstjes vinden en lezen, maar het nieuwe bestand niet maken. Welke stap van de taak heeft nog een tool nodig?",
            "all": "Dan kan de notitie hem nog steeds laten verwijderen en versturen, en moet de policy al die calls opvangen. Wat heeft de taak nodig, en niets meer?",
        },
    )
    choice_least
    return (choice_least,)


@app.cell(hide_code=True)
def text_paths():
    mo.vstack([
        mo.md(r"""
        ### Paden zijn verraderlijk

        Je `decide` vergelijkt het pad als tekst, en voor de calls van de run is dat genoeg. Echte paden hebben meer trucs dan `..`. Een link in `notes/`, een bestand dat voor een andere map staat, heeft een pad dat met `notes/` begint en toch daarbuiten uitkomt. En een policy die de map zonder schuine streep geeft, als `"notes"`, laat ook `notes-private/keys.txt` door, want dat pad begint met dezelfde letters. Echte code vraagt het besturingssysteem eerst waar een pad echt uitkomt, met `os.path.realpath`, en vergelijkt dat volledige pad met het volledige pad van de map.
        """),
        L.watch_out("Een controle in je code is één laag. Sterker is een laag waar de agent helemaal niet bij kan: laat hem draaien als een gebruiker van het besturingssysteem die alleen die ene map kan openen, of in een apart proces zonder netwerk. Een latere les voert code die een agent schreef uit in zo'n apart proces."),
    ])
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Tools die een agent goed kan gebruiken](/course/open/build-tool-design?lang=nl) vertelde het resultaat van een tool het model wat er misging; hier doet een weigering hetzelfde. De agent-loop van [Wat een agent is: een model, tools en een loop](/course/open/agent-idea?lang=nl) voert elke call uit die hij krijgt; met `run_call` tussen de loop en de tools komen alleen de calls door die je policy toestaat. De volgende les vraagt of een taak wel een agent nodig heeft, of een vaste reeks stappen die je code kiest.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_least, result_decide, result_run_call):
    L.wrap_up(result_decide, result_run_call, choice_least)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
