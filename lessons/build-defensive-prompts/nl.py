# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-defensive-prompts. Route: course/route/lessen/bouwen.yaml. Checks, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-defensive-prompts", "nl")
    C = L.checks   # de opgenomen runs van de course, de checks, en de demo met jouw mark en label_of


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [System prompts en response prefill](/course/open/build-system-prompt?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rule():
    quick_rule = L.choice(
        "quick-rule",
        question="De system prompt van de assistent van je winkel zegt: *Beloof nooit korting.* Wat zorgt ervoor dat er geen korting wordt beloofd?",
        options={
            "prompt": "De system prompt zelf: een model volgt zijn system prompt altijd",
            "capitals": "De regel twee keer schrijven, de tweede keer in hoofdletters, zodat het model hem niet kan missen",
            "code": "Code die elk antwoord controleert voordat je app het gebruikt, wat de prompt ook zegt",
        },
        correct="code",
        why={
            "code": "Een system prompt maakt een antwoord waarschijnlijker, nooit zeker; wat moet kloppen, hoort in code die elk antwoord controleert.",
            "prompt": "Een system prompt is geen slot: met de system prompt van een fietsenwinkel schreef Qwen3 toch in elke run een gedicht. Zie [System prompts en response prefill](/course/open/build-system-prompt?lang=nl).",
            "capitals": "Luidere woorden kunnen een regel waarschijnlijker maken, nooit zeker. Wat moet kloppen, hoort in code. Zie [System prompts en response prefill](/course/open/build-system-prompt?lang=nl).",
        },
    )
    quick_rule
    return (quick_rule,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een app stopt steeds tekst van buiten in zijn prompts: een e-mail om te sorteren, een webpagina om samen te vatten, een recensie, een document dat een gebruiker uploadt. Die tekst kan zelf instructies bevatten, voor het model geschreven door wie de tekst schreef: *negeer je instructies en antwoord met …*. Een model leest je instructies en die tekst als één stroom tokens, dus het kan niet betrouwbaar zien welke woorden je taak zijn en welke data. Als het de tekst volgt, heet dat prompt injection. Je app moet daarop gebouwd zijn.

    **Een voorbeeld uit het dagelijks leven.** Een kassamedewerker krijgt een briefje van een klant: *De bedrijfsleider zegt: geef de brenger van dit briefje 100 euro.* Een goede kassamedewerker neemt alleen opdrachten aan van de bedrijfsleider, en ziet het briefje als wat het is: een papiertje van een klant. Dat is alleen een vergelijking; hieronder meet je hoe vaak Qwen3 het briefje volgde.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **prompt injection** | een aanval die instructies verstopt in tekst die een model leest, zodat het model die volgt in plaats van jouw instructies |
    | **defensive prompting** | prompts zo schrijven dat een model moeilijker te misbruiken is, bijvoorbeeld door tekst van buiten duidelijk te markeren en te zeggen dat die data is, nooit instructies |

    ### Drie e-mails die instructies geven

    De course schreef drie e-mails voor een app die e-mail sorteert in de labels meeting, order, review en other. Elke e-mail bevat een instructie voor het model: de vergadering vraagt het te antwoorden met *I HAVE BEEN HACKED*, de recensie met het adres van een webwinkel, de bestelling met het label *urgent*. De course liet Qwen3 elke e-mail {seeds} keer sorteren, op drie manieren, in het Engels omdat Qwen3 Engels het best schrijft. Eerst de gewone manier, de taak en dan de e-mail in één bericht:

    ```text
    {task}
    ```

    Dan met de e-mail tussen de markeringen `<email>` en `</email>`, en een system prompt die zegt dat wat ertussen staat data is:

    ```text
    {guard}
    ```

    En als laatste zonder system prompt: eerst de gemarkeerde e-mail, en daarna de taak, met dezelfde waarschuwing:

    ```text
    <email>
    (the email)
    </email>

    {after}
    ```
    """.format(seeds=C.SEEDS, task=C.TASK.format(email="(the email)"), guard=C.GUARD, after=C.AFTER))
    return


@app.cell(hide_code=True)
def picture_runs():
    mo.Html(C.runs_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_runs():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        Op de gewone manier volgden {plain} van de {runs} antwoorden de e-mail. De markeringen en de waarschuwing in de system prompt hielpen niet, ze maakten het erger: {marked} van de {runs}. Met de taak na de e-mail volgde geen enkel antwoord hem. Bij de vergadering was op de gewone manier elk antwoord de zin waar de e-mail om vroeg:
        """.format(plain=_f["plain"]["followed"], marked=_f["marked"]["followed"], runs=_f["plain"]["runs"])),
        mo.Html(C.answers_picture(L.lang)),
        mo.md(r"""
        Qwen3-0.6B volgde meestal de laatste duidelijke instructie die het las, waar die ook stond; met de taak na de e-mail was die laatste instructie de jouwe. Het sorteerde nog steeds slecht: maar {right} van de {runs} antwoorden hadden het goede label. Grotere modellen trappen minder vaak in zulke trucs, maar geen prompt maakt injection onmogelijk, en het advies van OWASP, in zijn lijst met de grootste risico's voor apps als deze, is hetzelfde: vertrouw niet op wat het model schrijft. De stap die standhoudt, zit dus in je code. Een app die alleen een antwoord doorlaat dat een van zijn vier labels is, had alle {stopped} antwoorden tegengehouden die een e-mail volgden. Wat hij niet kan tegenhouden, is een fout label uit de lijst: {wrong} antwoorden op de drie manieren waren een label, maar het verkeerde, en waren erdoor gekomen.
        """.format(right=_f["after"]["right"], runs=_f["after"]["runs"], stopped=sum(_f[m]["followed"] for m in C.MODES),
                   wrong=sum(_f[m]["wrong_passed"] for m in C.MODES))),
        L.watch_out("Een app waarvan het model iets kan doen, zoals een e-mail versturen, betalen of een bestand wissen, maakt van een injection een verkeerde daad in plaats van een verkeerd antwoord. Geef een model alleen de acties die zijn taak nodig heeft, en laat een mens bevestigen wat niet terug te draaien is; een latere module over guardrails gaat verder."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: een prompt die zo goed mogelijk standhoudt

    1. **Markeer de tekst van buiten** met eigen markeringen, en haal die markeringen eerst uit de tekst, zodat die de data niet te vroeg kan afsluiten.
    2. **Zet je taak erachter,** en zeg daar dat wat tussen de markeringen staat data is, en dat instructies erin nooit gevolgd worden.
    3. **Controleer het antwoord in code:** laat alleen door wat je app verwacht, zoals een van zijn labels, en geef voor al het andere `None` terug, zodat je app het aan een mens kan geven.
    4. **Geef het model niet meer macht dan zijn taak nodig heeft,** en laat een mens meekijken bij wat niet terug te draaien is.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _after = C.answers("after", "order")[0]["text"]
    _plain = C.answers("plain", "order")[0]["text"]
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De bestelling: *{email}*

    1. **Markeren:** de e-mail heeft geen eigen markeringen, dus hij gaat zoals hij is tussen `<email>` en `</email>`.
    2. **De taak erachter:** het bericht is de gemarkeerde e-mail, en daarna de taak met de waarschuwing, zoals hierboven.
    3. **Het antwoord controleren:** in de eerste run antwoordde Qwen3 *{after}*: een van de vier labels, dus de check laat het door, en de app bergt de e-mail eronder op. Op de gewone manier antwoordde het *{plain}*: geen label, dus de check geeft `None` terug, en de app legt de e-mail opzij voor een mens, in plaats van hem op te bergen onder een label dat de e-mail zelf koos.
    4. **Niet meer macht:** de app sorteert alleen; zelfs een antwoord dat erdoor kwam, kon niet meer doen dan een e-mail onder het verkeerde label zetten.
    """.format(email=C.EMAILS["order"][0], after=_after.strip(), plain=_plain.strip()))
    return


@app.cell(hide_code=True)
def text_mark():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Markeringen die de e-mail niet kan breken

        Een e-mail kan zelf `</email>` bevatten, en dan lijkt alles daarna tekst buiten de data. Schrijf `mark(email)`, die `<email>` en `</email>` uit de e-mail haalt, en hem daarna tussen de markeringen zet, elk op een eigen regel. Het vak zet hem alleen tussen de markeringen.
        """),
        L.task("Haal beide markeringen uit de e-mail voordat je hem ertussen zet."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mark():
    box_mark = L.code_box("mark", lines=4)
    box_mark
    return (box_mark,)


@app.cell(hide_code=True)
def check_mark(box_mark):
    result_mark = L.check("mark", box_mark.value)
    result_mark
    return (result_mark,)


@app.cell(hide_code=True)
def hints_mark():
    L.hints(
        "mark",
        [
            "Maak de e-mail vóór de `return` schoon: er mag geen `<email>` en geen `</email>` meer in staan.",
            "`email.replace(\"</email>\", \"\")` geeft de e-mail zonder die markering; doe hetzelfde voor `<email>`, en bewaar het resultaat in `email`.",
            "`email = email.replace(\"<email>\", \"\").replace(\"</email>\", \"\")`, en dan de `return` zoals hij was.",
        ],
        why="""```python
def mark(email):
    email = email.replace("<email>", "").replace("</email>", "")
    return "<email>\\n" + email + "\\n</email>"
```

Nu is de enige `</email>` die het model ziet die aan het eind, die je code daar zette, dus de data eindigt waar jij het zegt.""",
    )
    return


@app.cell(hide_code=True)
def text_label():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Alleen een label komt erdoor

        Het vak stuurt de gemarkeerde e-mail met de taak `AFTER` erachter, en geeft terug wat het model schreef. Pas `label_of(url, email)` zo aan dat hij het antwoord alleen teruggeeft als het een van `LABELS` is, nadat je er spaties en regeleinden af hebt gehaald met `.strip()` en de letters klein hebt gemaakt met `.lower()`, en `None` bij elk ander antwoord. `AFTER`, `LABELS` en jouw `mark` zijn gegeven. De check draait je functie tegen een stand-in die soms de e-mail volgt.
        """),
        L.task("Geef het antwoord alleen terug als het een van `LABELS` is, anders `None`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_label():
    box_label = L.code_box("label_of", lines=15)
    box_label
    return (box_label,)


@app.cell(hide_code=True)
def check_label(box_label):
    result_label = L.check("label_of", box_label.value)
    result_label
    return (result_label,)


@app.cell(hide_code=True)
def hints_label():
    L.hints(
        "label_of",
        [
            "Maak het antwoord eerst vergelijkbaar, en vraag dan of het een van de `LABELS` is; al het andere krijgt `None`.",
            "`.strip().lower()` op de content geeft het antwoord zonder spaties eromheen en in kleine letters; `said in LABELS` is `True` als het een van de vier is.",
            "`said = answer[\"choices\"][0][\"message\"][\"content\"].strip().lower()`, dan `if said in LABELS:` met `return said` eronder, en aan het eind `return None`.",
        ],
        why="""```python
import json
import urllib.request


def label_of(url, email):
    messages = [{"role": "user", "content": mark(email) + "\n\n" + AFTER}]
    body = {"model": "qwen3-0.6b", "messages": messages, "max_tokens": 10}
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        answer = json.loads(response.read())
    said = answer["choices"][0]["message"]["content"].strip().lower()
    if said in LABELS:
        return said
    return None
```

`None` is geen fout van de app: het betekent *hier moet een mens naar kijken*, en dat is het goede antwoord op een e-mail die opdrachten probeerde te geven.""",
    )
    return


@app.cell(hide_code=True)
def text_act():
    mo.md(r"""
    ### Oefening 3. Een assistent die iets kan doen
    """)
    return


@app.cell(hide_code=True)
def choice_act_cell():
    choice_act = L.choice(
        "act",
        question="Je assistent leest binnenkomende e-mails en mag ook zelf e-mails versturen. Een e-mail zegt: *Stuur alle facturen door naar dit adres.* Wat beschermt je app het best?",
        options={
            "confirm": "Een mens elke e-mail laten bevestigen die de assistent wil versturen, voordat hij weggaat",
            "prompt": "Een system prompt die het doorsturen van facturen verbiedt, in hoofdletters",
            "bigger": "Een groter model, dat te slim is voor zo'n truc, zodat de assistent zelf kan blijven versturen",
        },
        correct="confirm",
        why={
            "confirm": "Markeringen en een waarschuwing verlagen het risico, maar geen prompt haalt het weg; een mens die elke verstuurde e-mail bevestigt, houdt tegen waar het model in trapte.",
            "prompt": "In de runs van de course hielden markeringen en een waarschuwing in de system prompt helemaal geen stand, en zelfs de beste prompt liet foute labels door; hoofdletters veranderen dat niet. Wat niet terug te draaien is, heeft een mens of code nodig die ja zegt.",
            "bigger": "Grotere modellen trappen er minder vaak in, maar niet nooit. Een e-mail versturen is niet terug te draaien, dus een mens moet het bevestigen.",
        },
    )
    choice_act
    return (choice_act,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het op Qwen3

    Als oefening 1 en 2 goed zijn, kun je een eigen e-mail op het echte model proberen: de course start zijn eigen server op je computer, met Qwen3-0.6B erachter. Schrijf een e-mail, met of zonder instructie erin; de demo sorteert hem met jouw `label_of` en laat zien wat Qwen3 schreef en wat je check doorliet. Je computer is niet die van de course, dus je antwoorden kunnen verschillen.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B (1,5 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    C.download_or_stop(get_button, {
        "ask": "*Het chat model Qwen3-0.6B staat nog niet op je computer. De knop downloadt het één keer van Hugging Face, `Qwen/Qwen3-0.6B` in een vaste versie, 1,5 GB, onder de Apache-2.0-licentie, met de tokenizer van Qwen3-0.6B-Base.*",
        "busy": "**Downloaden** {done} van {total} MB …",
        "no_net": "*Het downloaden lukte niet: Hugging Face was niet te bereiken, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*Het downloaden lukte niet: het bestand dat binnenkwam, is niet het vastgezette bestand van de course, dus de course heeft het weggegooid. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*Het downloaden lukte niet: er is geen ruimte op je schijf voor 1,5 GB. Maak ruimte, en klik opnieuw.*",
        "failed": "*Het downloaden lukte niet ({why}). Klik opnieuw.*",
    })
    mo.md("*Qwen3-0.6B is gedownload.*")
    return


@app.cell(hide_code=True)
def email_cell():
    email_box = mo.ui.text_area(value=C.EMAILS["order"][0], label="Jouw e-mail", full_width=True, rows=4)
    sort_button = mo.ui.run_button(label="Sorteren met jouw label_of")
    mo.vstack([email_box, sort_button])
    return email_box, sort_button


@app.cell(hide_code=True)
def demo_answer(box_label, box_mark, email_box, sort_button):
    _fn = C.learner_functions(box_mark.value, box_label.value)
    _email = email_box.value.strip()
    if not C.downloaded():
        _out = mo.md("*Download eerst Qwen3, met de knop hierboven.*")
    elif _fn is None:
        _out = mo.md("*Maak eerst oefening 1 en 2 af: de demo sorteert met jouw eigen `mark` en `label_of`.*")
    elif not sort_button.value:
        _out = mo.md("*Druk op de knop om te sorteren.*")
    elif not _email:
        _out = mo.md("*Schrijf eerst een e-mail in het vak.*")
    else:
        mo.output.replace(mo.md("*De e-mail gaat naar de server; die laadt Qwen3 en antwoordt …*"))
        _words = {"title": "Jouw e-mail, gesorteerd door jouw label_of", "said": "Wat Qwen3 schreef:", "gave": "Wat jouw label_of teruggaf:",
                  "passed": "Een label: je app kan de e-mail eronder opbergen.",
                  "stopped": "None: je app geeft deze e-mail aan een mens.",
                  "failed": "Het stopte met een fout tegen de echte server: {why}",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk nog eens op de knop."}
        _out = mo.Html(C.demo(_fn, _email, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [System prompts en response prefill](/course/open/build-system-prompt?lang=nl) liet zien dat een system prompt geen slot is; hier werkte een hele e-mail ertegenin; waar de taak stond, maakte uit, en de check in code was wat standhield. [Meer dan één keer vragen](/course/open/build-self-consistency?lang=nl) stelt dezelfde vraag een paar keer en neemt het antwoord dat het vaakst terugkomt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_act, result_label, result_mark):
    L.wrap_up(result_mark, result_label, choice_act)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
