# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-context-engineering. Route: course/route/lessen/bouwen.yaml. Checks, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-context-engineering", "nl")
    C = L.checks   # de opgenomen runs van de course, de checks, en de demo met je compact


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
def quick_check_system():
    quick_system = L.choice(
        "quick-system",
        question="Je app heeft een system prompt. Hoe vaak gaat die naar het model?",
        options={
            "once": "Eén keer, aan het begin van het gesprek: daarna onthoudt het model hem",
            "first": "Met het eerste request van elke dag, als de server opnieuw start",
            "every": "Met elk request, aan het begin van de berichten, vóór het gesprek",
        },
        correct="every",
        why={
            "every": "Het model onthoudt niets tussen requests, dus de system prompt gaat er elke keer opnieuw in, en kost elke keer input tokens.",
            "once": "Het model onthoudt niets tussen requests: de app stuurt de system prompt met elk request mee. Zie [System prompts en response prefill](/course/open/build-system-prompt?lang=nl).",
            "first": "Het model onthoudt niets tussen requests, ook niet binnen een dag: de app stuurt de system prompt met elk request mee. Zie [System prompts en response prefill](/course/open/build-system-prompt?lang=nl).",
        },
    )
    quick_system
    return (quick_system,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een model weet niets van je gesprek behalve wat er in het request staat. Een chatapp stuurt dus bij elk nieuw bericht het hele gesprek opnieuw, en het gesprek groeit alleen maar. Elk request kost dan meer input tokens en duurt langer om te lezen; op een gegeven moment past het niet meer in het context window, en lang daarvoor al vinden modellen een feit midden in een lange tekst slechter terug dan aan het begin of het eind, zoals het onderzoek *Lost in the Middle* in 2023 mat. De app moet kiezen wat erin gaat, en die keuze bepaalt wat het model kan weten.

    **Een voorbeeld uit het dagelijks leven.** Een arts die je voor het eerst ziet, leest niet elke notitie van twintig jaar bezoeken. Het dossier heeft een samenvatting, en een allergie staat in rood op de voorkant, waar niemand hem kan missen. Dat is maar een vergelijking; hieronder meet je wat Qwen3 nog weet als het begin van een gesprek wordt weggelaten, en wat het terugbrengt.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **context engineering** | bij elk request kiezen wat er in het context window van het model gaat: instructies, de berichten die ertoe doen, notities, documenten, en weinig meer |
    | **compaction** | een lang gesprek korter maken als het te lang wordt: de oude berichten worden vervangen door een samenvatting ervan, zodat het werk door kan gaan |

    ### Wat het model te zien krijgt

    De course schreef een gesprek over een etentje van {n} berichten, vijf van de gebruiker en vijf antwoorden. Het eerste bericht bevat een feit dat aan het eind ertoe doet: *{first}* Daarna gaat het gesprek over het voorgerecht, het hoofdgerecht, de wijn en de muziek. De laatste vraag was:

    > {last}

    De course stelde hem {seeds} keer op elk van vier manieren: met het hele gesprek; met een venster van alleen de laatste {keep} berichten; met dat venster en een notitie in de system prompt, *{note}*; en met dat venster en een samenvatting van de berichten die wegvielen, die Qwen3 zelf schreef, aan het eind van de system prompt. Een antwoord telt als het de notenallergie noemt:
    """.format(n=len(C.TALK), first=C.TALK[0][1], last=C.LAST, seeds=C.SEEDS, keep=C.KEEP, note=C.NOTE))
    return


@app.cell(hide_code=True)
def picture_modes():
    mo.Html(C.modes_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_modes():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        Met het hele gesprek noemden {whole} van de {seeds} antwoorden de notenallergie; met het venster {window}: het eerste bericht zat niet in het request, dus voor het model bestond de allergie niet, en de eerste run verzon er een andere regel voor in de plaats: *{made_up}* De notitie bracht de allergie terug, {notes} van de {seeds}, en de samenvatting ook, {compact} van de {seeds}. Maar de notitie bevatte alleen de allergie, dus geen enkel antwoord ermee noemde de zes vrienden; de samenvatting bevatte allebei, en {compact_six} van de {seeds} antwoorden ermee noemden ze, tegen {whole_six} met het hele gesprek. Het venster en de samenvatting kostten ook minder tokens: het request met het hele gesprek had {whole_tokens} prompt tokens, dat met het venster {window_tokens}, en dat met de samenvatting {compact_tokens}. De samenvatting kostte zelf een aanroep, {sp} prompt tokens en {so} geschreven tokens; dat verdient zich terug zodra je hem met de requests daarna meestuurt.
        """.format(made_up=C.answers("window")[0]["text"].strip(), seeds=C.SEEDS, whole=_f["whole"]["knows"], window=_f["window"]["knows"], notes=_f["notes"]["knows"],
                   compact=_f["compact"]["knows"], compact_six=_f["compact"]["six"], sp=_f["summary_cost"]["prompt_tokens"], so=_f["summary_cost"]["tokens"], whole_six=_f["whole"]["six"], whole_tokens=_f["whole"]["prompt_tokens"],
                   window_tokens=_f["window"]["prompt_tokens"], compact_tokens=_f["compact"]["prompt_tokens"])),
        L.watch_out("Een samenvatting houdt wat het model dat hem schreef belangrijk vond, en verder niets. Wat nooit verloren mag gaan, zoals een allergie, een budget of een deadline, hoort in een notitie die je code bewaart en elke keer meestuurt, niet alleen in een samenvatting die een model steeds opnieuw schrijft."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: wat er in het venster gaat

    1. **Wat altijd geldt, gaat in de system prompt:** de instructies, en notities over de gebruiker die nooit verloren mogen gaan.
    2. **Het gesprek gaat erin als venster:** de laatste berichten, waarop het volgende antwoord voortbouwt.
    3. **Wat wegvalt, krijgt een samenvatting** (compaction), in de system prompt, zodat de kern van het begin blijft.
    4. **De rest wordt pas opgehaald als het nodig is,** zoals een document waar de vraag over gaat: dat doet de volgende module.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _f = C.findings()
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Hoe bracht de samenvatting de allergie terug?

    1. **Wat altijd geldt:** de system prompt is *{system}*
    2. **Het venster:** de laatste {keep} berichten, over de wijn en de muziek, en daarna de vraag.
    3. **De samenvatting:** de course vroeg Qwen3 de {dropped} berichten samen te vatten die wegvielen, met de prompt in het vak hieronder. Het schreef: *{summary}* Die tekst kwam aan het eind van de system prompt, na *Summary of the conversation so far:*.
    4. **Verder niets:** het request had {tokens} prompt tokens, tegen {whole_tokens} voor het hele gesprek.

    ```text
    {summarize}
    ```
    """.format(system=C.SYSTEM, keep=C.KEEP, dropped=len(C.TALK) - C.KEEP, summary=_f["summary"].strip(),
               tokens=_f["compact"]["prompt_tokens"], whole_tokens=_f["whole"]["prompt_tokens"], summarize=C.SUMMARIZE))
    return


@app.cell(hide_code=True)
def text_window():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een venster dat de system prompt houdt

        Schrijf `window(messages, keep)`, die de system prompt teruggeeft, `messages[0]`, met daarachter de laatste `keep` berichten van het gesprek. De system prompt moet precies één keer meekomen, ook als `keep` tot het begin van het gesprek terugreikt, en met `keep` 0 blijft alleen de system prompt over. Het vak houdt de laatste `keep` berichten, maar laat de system prompt weg. `max(a, b)` geeft het grootste van twee getallen.
        """),
        L.task("Zet `messages[0]` vooraan, en laat de slice bij 1 of later beginnen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_window():
    box_window = L.code_box("window", lines=6)
    box_window
    return (box_window,)


@app.cell(hide_code=True)
def check_window(box_window):
    result_window = L.check("window", box_window.value)
    result_window
    return (result_window,)


@app.cell(hide_code=True)
def hints_window():
    L.hints(
        "window",
        [
            "Twee veranderingen: de lijst begint met de system prompt, en de slice mag nooit terugreiken tot plaats 0, waar de system prompt staat.",
            "Begin met `out = [messages[0]]`. De slice begint bij `len(messages) - keep`, maar nooit onder 1; `max` geeft het grootste van twee getallen.",
            "Na `out = [messages[0]]`: `for message in messages[___:]:`, met de `append` zoals hij was.",
        ],
        why="""```python
def window(messages, keep):
    out = [messages[0]]
    for message in messages[max(1, len(messages) - keep):]:
        out.append(message)
    return out
```

`messages[-keep:]` lijkt korter, maar met `keep` 0 is het `messages[-0:]`, en dat is `messages[0:]`: de hele lijst. Tellen vanaf het begin heeft die verrassing niet.""",
    )
    return


@app.cell(hide_code=True)
def text_compact():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een samenvatting van wat wegvalt

        Het vak vraagt de server om een samenvatting van de berichten die uit het venster vallen, `old`, met de prompt `SUMMARIZE` om een `transcript` ervan, en geeft daarna het venster zonder de samenvatting terug. Verander het eind van `compact(url, messages, keep)` zo dat het het venster teruggeeft met een nieuw system-bericht vooraan: de content van `messages[0]`, dan `"\n\nSummary of the conversation so far: "`, dan de samenvatting. Maak er een nieuwe dictionary voor, zodat de lijst berichten die je kreeg hetzelfde blijft. `SUMMARIZE`, `transcript` en `window` zijn gegeven. De check draait je functie tegen een stand-in.
        """),
        L.task("Geef het venster terug met een nieuw system-bericht: de system prompt, de regel, en de samenvatting."),
    ])
    return


@app.cell(hide_code=True)
def exercise_compact():
    box_compact = L.code_box("compact", lines=18)
    box_compact
    return (box_compact,)


@app.cell(hide_code=True)
def check_compact(box_compact):
    result_compact = L.check("compact", box_compact.value)
    result_compact
    return (result_compact,)


@app.cell(hide_code=True)
def hints_compact():
    L.hints(
        "compact",
        [
            "Alleen de laatste regel verandert: het venster houdt zijn berichten, maar het eerste, de system prompt, krijgt de samenvatting aan het eind.",
            "Maak `out = window(messages, keep)`, en een nieuwe dictionary met de rol `system` en als content `messages[0][\"content\"] + \"\\n\\nSummary of the conversation so far: \" + summary`. In de lijst die je teruggeeft, neemt die de plaats in van het eerste bericht van `out`.",
            "Als laatste regel: `return [system] + ___`.",
        ],
        why="""```python
import json
import urllib.request


def compact(url, messages, keep):
    old = messages[1:max(1, len(messages) - keep)]
    body = {"model": "qwen3-0.6b", "max_tokens": 120,
            "messages": [{"role": "user", "content": SUMMARIZE.format(talk=transcript(old))}]}
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        answer = json.loads(response.read())
    summary = answer["choices"][0]["message"]["content"]
    out = window(messages, keep)
    system = {"role": "system",
              "content": messages[0]["content"] + "\\n\\nSummary of the conversation so far: " + summary}
    return [system] + out[1:]
```

`out[1:]` is het venster zonder zijn eerste bericht, de oude system prompt; de nieuwe neemt zijn plaats in. Als je `messages[0]["content"]` zelf verandert, verander je het gesprek dat je app bewaart, en stapelen de samenvattingen zich daar op bij elke aanroep.""",
    )
    return


@app.cell(hide_code=True)
def text_notes():
    mo.md(r"""
    ### Oefening 3. Een feit dat moet blijven
    """)
    return


@app.cell(hide_code=True)
def choice_notes_cell():
    choice_notes = L.choice(
        "notes",
        question="Een gebruiker van je kookapp zegt in het eerste bericht geen gluten te kunnen eten. Waar bewaart je app dat, zodat het nooit verloren gaat?",
        options={
            "first": "In het eerste bericht van het gesprek, waar de gebruiker het schreef, zodat het in de geschiedenis blijft",
            "model": "Nergens: als het model het gelezen heeft, weet het dat",
            "note": "Als notitie die je code bewaart en in de system prompt van elk request zet",
        },
        correct="note",
        why={
            "note": "De system prompt gaat met elk request mee, wat het venster ook houdt, dus de notitie staat altijd in het context window.",
            "first": "Het eerste bericht valt als eerste uit een venster, en een samenvatting kan het weglaten. Bewaar het als notitie in de system prompt, die elke keer meegaat.",
            "model": "Het model onthoudt niets tussen requests: wat niet in het request staat, bestaat niet voor het model, zoals het venster liet zien. Bewaar het als notitie in de system prompt.",
        },
    )
    choice_notes
    return (choice_notes,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het op Qwen3

    Zodra oefening 1 en 2 goed zijn, kun je je `compact` met het echte model op het gesprek over het etentje draaien: de course start zijn eigen server op je computer, met Qwen3-0.6B erachter. De demo maakt het venster met het aantal berichten dat je kiest, laat Qwen3 de rest samenvatten, stelt de laatste vraag met wat je `compact` teruggaf, en laat de samenvatting en het antwoord zien. Je computer is niet die van de course, dus je antwoorden kunnen verschillen.
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
def keep_cell():
    keep_slider = mo.ui.slider(start=0, stop=8, step=2, value=4, label="Berichten in het venster", show_value=True)
    compact_button = mo.ui.run_button(label="Vraag met je compact")
    mo.vstack([keep_slider, compact_button])
    return compact_button, keep_slider


@app.cell(hide_code=True)
def demo_answer(box_compact, box_window, compact_button, keep_slider):
    _fn = C.learner_compact(box_window.value, box_compact.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif _fn is None:
        _out = mo.md("*Maak eerst oefening 1 en 2 af: de demo draait je eigen `compact` en `window`.*")
    elif not compact_button.value:
        _out = mo.md("*Druk op de knop om te vragen.*")
    else:
        mo.output.replace(mo.md("*Qwen3 vat samen en antwoordt dan; de server laadt het en schrijft …*"))
        _words = {"title": "Het gesprek over het etentje met je compact, {keep} berichten in het venster".format(keep=keep_slider.value),
                  "summary": "De samenvatting:", "answer": "Het antwoord op de laatste vraag:",
                  "knows": "Het noemde de notenallergie.", "forgot": "Het noemde de notenallergie niet.",
                  "tokens": "{tokens} prompt tokens, in {seconds} seconden.",
                  "failed": "Het stopte met een fout tegen de echte server: {why}",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.demo(_fn, keep_slider.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De system prompt van [System prompts en response prefill](/course/open/build-system-prompt?lang=nl) is de plaats voor wat altijd geldt, en nu ook voor notities en een samenvatting. Alles wat in het venster gaat, wordt bij elk request opnieuw gelezen; [Eén keer betalen voor hetzelfde begin](/course/open/build-prompt-caching?lang=nl) maakt het deel dat hetzelfde blijft goedkoop om opnieuw te lezen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_notes, result_compact, result_window):
    L.wrap_up(result_window, result_compact, choice_notes)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
