# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les agent-idea. Route: course/route/lessen/bouwen.yaml. Checks, de stand-ins en de runs van de course: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("agent-idea", "nl")
    C = L.checks   # de voorraadkast, zijn twee tools, de stand-ins voor het model, en de runs van Qwen3 van de course
    R = C.course()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Drie korte vragen over [Leren van rewards](/course/open/rl-what-is-rl?lang=nl), [Prompts die zichzelf controleren](/course/open/start-prompt-chains?lang=nl) en [Data als tekst: JSON](/course/open/py-json?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rl():
    quick_rl = L.choice(
        "quick-rl",
        question="Een robotstofzuiger leert met reinforcement learning. Wat geeft de environment na elke zet terug aan de agent?",
        options={
            "policy": "Zijn policy: welke zet hij hierna doet",
            "both": "De nieuwe situatie, en een reward",
            "return": "De return van de hele run",
        },
        correct="both",
        why={
            "policy": "De policy hoort bij de agent: zo kiest de agent. Zie de nieuwe woorden in [Leren van rewards](/course/open/rl-what-is-rl?lang=nl).",
            "both": "Na elke action geeft de environment terug waar de agent nu is, en een getal voor hoe goed die stap was.",
            "return": "De return telt de rewards van veel stappen op; na één zet is er één reward. Zie [Leren van rewards](/course/open/rl-what-is-rl?lang=nl).",
        },
    )
    quick_rl
    return (quick_rl,)


@app.cell(hide_code=True)
def quick_check_chain():
    quick_chain = L.choice(
        "quick-chain",
        question="Je laat een model in drie prompts een samenvatting van een lang rapport schrijven: eerst de hoofdpunten, dan een concept uit die punten, dan een kortere versie van het concept. Hoe heet dat?",
        options={
            "few-shot": "Few-shot prompting",
            "chaining": "Prompt chaining",
            "refine": "Self-Refine",
        },
        correct="chaining",
        why={
            "few-shot": "Few-shot prompting zet voorbeelden in één prompt, zoals in [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl). Hier gaat het antwoord van elke prompt naar de volgende. Zie [Prompts die zichzelf controleren](/course/open/start-prompt-chains?lang=nl).",
            "chaining": "Kleine prompts na elkaar, waarbij elk antwoord in de volgende prompt gaat.",
            "refine": "Bij Self-Refine beoordeelt het model zijn eigen antwoord en verbetert het dat. Hier doet elke prompt een andere stap. Zie [Prompts die zichzelf controleren](/course/open/start-prompt-chains?lang=nl).",
        },
    )
    quick_chain
    return (quick_chain,)


@app.cell(hide_code=True)
def quick_check_json():
    quick_json = L.choice(
        "quick-json",
        question="Wat geeft dit?\n\n```python\nimport json\njson.loads('{\"item\": \"milk\", \"grams\": 200}')[\"grams\"]\n```",
        options={
            "number": "`200`, een getal",
            "text": "`\"200\"`, als tekst",
            "error": "Een fout: `\"grams\"` is geen key",
        },
        correct="number",
        why={
            "text": "In de JSON-tekst staat 200 zonder aanhalingstekens, dus het is een getal, en `json.loads` geeft het getal. Zie [Data als tekst: JSON](/course/open/py-json?lang=nl).",
            "number": "`json.loads` maakt van de tekst een dict, en de waarde bij de key `\"grams\"` is het getal 200.",
            "error": "`json.loads` geeft een dict met de keys `\"item\"` en `\"grams\"`. Zie [Data als tekst: JSON](/course/open/py-json?lang=nl).",
        },
    )
    quick_json
    return (quick_json,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Vraag een chat model wat je moet kopen om pannenkoeken te bakken, en het kan het niet weten: het kan niet in je kast kijken. Een language model schrijft alleen tekst. Geef hetzelfde model twee functies die het mag gebruiken, één die een recept opzoekt en één die zegt hoeveel je van een ingrediënt in je voorraadkast hebt, en een klein programma dat ze voor het model uitvoert, en het kan het uitzoeken. Dat programma met het model is een **agent**. De coding assistants die je bestanden lezen en je tests draaien, en de chatassistenten die op het web zoeken voordat ze antwoorden, zijn diezelfde loop om een language model heen. In deze les schrijf je die loop zelf, in een stuk of twaalf regels.
    """)
    return


@app.cell(hide_code=True)
def predict_no_tools():
    guess_no_tools = L.guess(
        "no-tools",
        question=f"Qwen3-0.6B, het chat model van de course, krijgt de Engelse vraag *{R['task']}* (ik wil pannenkoeken bakken: welke ingrediënten moet ik kopen, en hoeveel gram van elk?), en geen tools. De course stelde de vraag {len(R['without_tools'])} keer, elke keer met een andere seed. Wat deed het?",
        options={
            "asks": "Elke keer zei het dat het je voorraadkast niet kan zien, en vroeg het wat je hebt",
            "both": "Soms vroeg het wat je hebt, andere keren verzon het een antwoord",
            "invents": "Elke keer verzon het een antwoord",
        },
        reveal=C.no_tools_reveal(L.lang),
    )
    guess_no_tools
    return (guess_no_tools,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Het model voert zelf nooit iets uit: het kan alleen tekst schrijven. Dus vertelt je programma het in de prompt welke tools er zijn: de naam van elke tool, wat die doet, en de argumenten. Om er één te gebruiken, schrijft het model een **tool call** in een vaste vorm, als JSON: de naam van de tool en de argumenten, zoals `{"name": "pantry", "arguments": {"item": "milk"}}`. Je programma leest dat, voert de functie uit, en zet wat die teruggaf in het gesprek, als een eigen bericht. Dan vraagt je programma het model opnieuw. Het model leest het resultaat, en schrijft de volgende tool call, of zijn antwoord. Zo gaat het steeds rond: dat is de **agent loop**. Omdat een model tools kan blijven aanroepen, vraagt de loop het hooguit een vast aantal keer: de **stappenlimiet**.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **LLM agent**, of agent | een language model dat in een loop naar een doel werkt: het kiest een tool, ziet het resultaat, en kiest opnieuw, tot het klaar is | `run_agent(ask, tools, task, max_steps)` |
    | **tool** | een functie die je programma het model laat gebruiken, met een naam, een beschrijving en argumenten; ze geeft tekst terug | `tools["pantry"]({"item": "milk"})` |
    | **tool call** | wat het model schrijft om een tool te gebruiken: de naam van de tool en zijn argumenten, als JSON | `{"name": "pantry", "arguments": {"item": "milk"}}` |
    | **agent loop** | vraag het model, voer de tools uit die het aanroept, voeg hun resultaten toe aan de berichten, en vraag opnieuw, tot het antwoordt zonder tool call | `for step in range(max_steps):` |
    | **stappenlimiet** | het grootste aantal keer dat de loop het model vraagt, ook als het tools blijft aanroepen | `max_steps` |
    | **autonomiespectrum** | van een chatbot die alleen antwoordt tot een agent die een hele taak zelf doet: hoeveel het model beslist zonder jou | |
    """)
    return


@app.cell(hide_code=True)
def watch_out_words():
    mo.vstack([
        L.watch_out("*Agent* betekent hier iets anders dan in [Leren van rewards](/course/open/rl-what-is-rl?lang=nl). Daar leert een agent een policy uit rewards, door te proberen. Hier is een LLM agent een language model met tools: niemand geeft het rewards terwijl het werkt, en wat het hierna doet, komt van het model, dat de berichten las. Wat ze delen: iets dat handelt, ziet wat er gebeurde, en weer handelt, naar een doel. In deze les is *agent* de LLM agent, behalve waar *RL agent* staat."),
        L.watch_out("*Tool* betekent hier iets anders dan in [Welke AI-tool waarvoor](/course/open/start-ai-tools-landscape?lang=nl). Daar was een AI-tool een heel programma, zoals een chatassistent. Hier is een tool één functie die het model je programma mag vragen uit te voeren."),
    ])
    return


@app.cell(hide_code=True)
def picture_loop():
    mo.vstack([
        draw.flow(
            [("Het language model", ["leest de berichten", "schrijft een tool call", "of het antwoord"]),
             ("Je programma", ["de agent loop:", "houdt de berichten bij", "voert de tools uit"]),
             ("De tools", ["recipe(dish)", "pantry(item)"])],
            ["een tool call,\nof het antwoord", "de call,\nmet de argumenten"], ["alle berichten\ntot nu toe", "het resultaat,\nals tekst"],
            title="Een agent: een model, tools en een loop"),
        mo.md(r"""
        Het model en de tools komen elkaar nooit tegen: je programma zit ertussen. Het stuurt het model elke keer alle berichten tot nu toe, omdat het model tussen twee vragen niets onthoudt. Schrijft het model een tool call, dan voert je programma de tool uit en voegt het het resultaat toe aan de berichten. Schrijft het model een antwoord zonder tool call, dan stopt de loop.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_prompt():
    mo.vstack([
        mo.md(r"""
        **Wat het model echt leest.** De tools gaan als tekst in de prompt, in het eerste bericht, met de rol `system`: het bericht met de instructies van wie het model draait. Voor de voorraadkast is dit de tekst die Qwen3 leest vóór zijn eerste antwoord, zoals de course die stuurt, in het Engels. `<|im_start|>` en `<|im_end|>` geven aan waar elk bericht begint en eindigt, met de rol direct na het begin:
        """),
        mo.md("```text\n" + C.prompt_example() + "\n```"),
        mo.md(r"""
        Alles wat het model over de tools weet, staat in die tekst: hun namen, wat ze doen, hun argumenten, en de vorm van een tool call, tussen `<tool_call>`-tags. Qwen3 leerde van veel gesprekken in precies deze vorm, dus schrijft het zijn tool calls ook zo. De lege `<think>` en `</think>` helemaal aan het eind zeggen Qwen3 dat het antwoordt zonder eerst zijn redenering uit te schrijven; zo blijft het snel op een laptop. De `ask` van de course vindt de tool calls in het antwoord, leest hun JSON met `json.loads`, en geeft je loop elke call als een dict, met zijn `"name"` en zijn `"arguments"`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_spectrum():
    mo.md(r"""
    **Van chatbot naar agent.** Hoeveel beslist het model zelf? Dat is het autonomiespectrum:

    | | Wie kiest de volgende stap | Voorbeeld |
    |---|---|---|
    | een chatbot | jij: hij antwoordt alleen | een chatassistent zonder tools |
    | een assistent met tools | het model stelt een stap voor, jij zegt ja of nee | een coding assistant die het eerst vraagt voordat hij een opdracht uitvoert |
    | een agent | het model, binnen zijn stappenlimiet en zijn tools | een coding assistant die zelf bestanden mag aanpassen en tests mag draaien tot ze slagen |

    Hoe verder, hoe meer een agent zonder jou gedaan krijgt, en hoe meer er mis kan gaan voordat iemand het ziet. Een tool die alleen leest, zoals `pantry`, kan weinig kwaad; een tool die een e-mail verstuurt of een bestand verwijdert, wel. Daarom krijgen agents alleen de tools die ze nodig hebben, en vragen ze het eerst bij een stap die niet terug te draaien is: een les van het domein *Bouwen met AI* gaat daarover.

    **De RL agent en de LLM agent.** Allebei zijn ze dezelfde loop: kijk naar de situatie, handel, zie wat er gebeurde, handel weer. Ze verschillen in waar de keuze vandaan komt.

    | | RL agent | LLM agent |
    |---|---|---|
    | ziet | de situatie in de environment | alle berichten tot nu toe: de taak en elk resultaat van een tool |
    | doet | een action | een tool call, of het antwoord |
    | kiest met | een policy die hij leerde uit rewards, door te proberen | een language model, getraind op tekst, dat de berichten leest |
    | stopt | als de taak eindigt, zoals een spel dat gewonnen of verloren is | als hij antwoordt zonder tool call, of bij de stappenlimiet |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een language model met tools aan een taak te laten werken:

    1. Geef elke tool één argument, een dict met zijn argumenten, en laat hem tekst teruggeven.
    2. Begin de berichten met de taak: `[{"role": "user", "content": task}]`.
    3. Vraag het model om zijn volgende bericht, met alle berichten tot nu toe, en voeg zijn bericht eraan toe.
    4. Roept het geen tool aan, dan heeft het geantwoord: je bent klaar.
    5. Voer anders elke tool uit die het aanroept, met zijn argumenten, en voeg elk resultaat toe als bericht met de rol `"tool"`.
    6. Ga terug naar stap 3, maar hooguit `max_steps` keer.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het model is hier een stand-in, `ask`: een functie die de berichten leest en antwoordt zoals een model dat zijn tools goed gebruikt, altijd hetzelfde, zodat je elke stap ziet. Ze geeft een bericht terug als dict: `"content"` is de tekst, en `"tool_calls"` is een lijst van de tools die het model aanroept, elk met zijn `"name"` en zijn `"arguments"`. De tools staan in de dict `tools`. De taak staat in het Engels, zoals de stand-in hem leest: wat moet ik kopen om pannenkoeken te bakken? De cel doet stap 2 tot en met 5 van het recept met de hand, ronde na ronde, tot het model antwoordt. Voer hem uit.
    """)
    return


@app.cell
def example_kitchen():
    ask, tools = C.kitchen()   # de keuken (kitchen): de stand-in voor het model, en de twee tools

    # 2. de taak is het eerste bericht
    _messages = [{"role": "user", "content": "What do I have to buy to make pancakes?"}]
    # 3. vraag het model, en voeg zijn bericht toe
    _reply = ask(_messages)
    _messages.append(_reply)
    print(_reply["tool_calls"])
    # 5. voer elke tool uit die het aanroept, en voeg elk resultaat toe als bericht
    for _call in _reply["tool_calls"]:
        _result = tools[_call["name"]](_call["arguments"])
        _messages.append({"role": "tool", "content": _result})
        print(_result)
    # 3. en 5. opnieuw: het model ziet nu het recept
    _reply = ask(_messages)
    _messages.append(_reply)
    print(_reply["tool_calls"])
    for _call in _reply["tool_calls"]:
        _messages.append({"role": "tool", "content": tools[_call["name"]](_call["arguments"])})
    # 3. en 4. een derde keer: het roept geen tool aan, dus dit is zijn antwoord
    _reply = ask(_messages)
    print(_reply["tool_calls"], _reply["content"])
    return ask, tools


@app.cell(hide_code=True)
def text_after_example():
    mo.md(rf"""
    Onder de cel verschijnen vier regels. Het eerste bericht van het model roept één tool aan: `recipe`, met `{{"dish": "pancakes"}}`. De tool geeft het recept terug als JSON-tekst: `{{"flour": 250, "milk": 450, "eggs": 110, "butter": 25}}`, in gram. Met dat recept in de berichten roept het volgende bericht van het model `pantry` vier keer tegelijk aan, één keer voor elk ingrediënt. De cel voert die vier uit, voegt hun resultaten toe, en vraagt het een derde keer. Nu roept het model geen tool aan, `[]`, en is zijn bericht het antwoord: *Buy 250 grams of milk and 110 grams of eggs.* (koop 250 gram melk en 110 gram eieren). Dat is stap 4: de taak is klaar. Stap 3 en 5 kwamen terug, elke keer met andere tool calls: daar is een loop voor.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_loop():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De agent loop

        Schrijf `run_agent(ask, tools, task, max_steps)`: die begint de berichten met de taak, vraagt `ask` hooguit `max_steps` keer om het volgende bericht van het model, voert de tools uit die dat bericht aanroept, en geeft de hele lijst berichten terug. Drie regels zijn voor jou, bij de `...`. De check probeert je loop met de stand-in van het uitgewerkte voorbeeld, en met drie andere stand-ins: een die alleen praat, een die twee tools tegelijk aanroept, en een die nooit ophoudt met tools aanroepen.
        """),
        L.task("Laat `run_agent(ask, tools, \"What do I have to buy to make pancakes?\", 5)` de 9 berichten van de pannenkoeken geven: de taak, dan het model, een tool, het model, vier tools, en het antwoord van het model."),
    ])
    return


@app.cell(hide_code=True)
def exercise_loop():
    box_loop = L.code_box("run-agent", lines=11)
    box_loop
    return (box_loop,)


@app.cell(hide_code=True)
def check_loop(box_loop):
    result_loop = L.check("run-agent", box_loop.value)
    result_loop
    return (result_loop,)


@app.cell(hide_code=True)
def hints_loop():
    L.hints(
        "run-agent",
        [
            "Het uitgewerkte voorbeeld deed de stappen met de hand, de ene ronde na de andere; de loop doet stap 3 tot en met 5 steeds opnieuw. Welke regel van het voorbeeld vraagt het model, welke voert een tool uit, en wanneer stopte het?",
            "Het model heeft alle berichten tot nu toe nodig, dus het krijgt de lijst `messages`. Roept zijn bericht geen tool aan, dan is `reply[\"tool_calls\"]` een lege lijst, en is de taak klaar. Een tool krijgt `call[\"arguments\"]`.",
            "`reply = ask(messages)`, dan onder de `if`: `return ___`, en in de binnenste loop `result = tools[___](call[\"arguments\"])`.",
        ],
        why="`ask(messages)` geeft het model elk bericht tot nu toe, zijn eigen eerdere tool calls en elk resultaat inbegrepen. Een antwoord zonder tool call maakt de taak af, dus `return messages` geeft de lijst daar meteen terug. Anders voert de binnenste loop elke tool uit bij zijn naam, met zijn argumenten. En `range(max_steps)` vraagt het model hooguit `max_steps` keer: de stand-in die steeds de tijd vraagt, wordt na 3 vragen gestopt.",
    )
    return


@app.cell(hide_code=True)
def text_live():
    mo.md(rf"""
    ### Je loop met een echt model

    Je `run_agent` werkt met elke `ask` die berichten leest en een bericht teruggeeft. Hier krijgt hij Qwen3-0.6B, met de twee tools van de voorraadkast en de instructies van de course in een bericht met de rol `system` aan het begin, in het Engels: *{R['system']}* (je bent een keukenassistent met twee tools; zoek ermee het recept op en hoeveel er van elk ingrediënt in de voorraadkast is; vraag het niet aan de gebruiker; weet je genoeg, antwoord dan met wat er gekocht moet worden). De course voerde dit {len(R['runs'])} keer uit, met de `run_agent` van de course, elke keer met een andere seed; hieronder loop je die runs stap voor stap door. {C.runs_summary(L.lang)}
    """)
    return


@app.cell(hide_code=True)
def course_runs_cell():
    L.explore("course-runs", "run", lambda _v: C.run_frame(R["runs"][int(_v) - 1], int(_v), len(R["runs"]), L.lang), start=C.SHOWN_RUN, low=1,
              high=len(R["runs"]), step=1, role="", task="Loop de runs door. Welke tool roept het model nooit aan, en wat zegt zijn antwoord toch over de voorraadkast?")
    return


@app.cell(hide_code=True)
def text_runs_after():
    mo.md(C.runs_after(L.lang))
    return


@app.cell(hide_code=True)
def text_download():
    mo.md(rf"""
    **Voer het zelf uit.** Met de knop draait je eigen `run_agent` met Qwen3 op dezelfde taak, met de seed van run {C.SHOWN_RUN}: op de computer van de course geeft dat run {C.SHOWN_RUN} opnieuw, en op de jouwe kan afronding een token veranderen, en vanaf daar kan de run een andere kant op gaan. Op de computer van de course, met één thread, duurde het {C.number(C.seconds_of(R['runs'][C.SHOWN_RUN - 1]), 0, L.lang)} seconden; met meer threads gaat het sneller. Terwijl Qwen3 schrijft, zie je zijn bericht tot nu toe.
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
def run_button_cell():
    run_button = mo.ui.run_button(label="Voer Qwen3 uit met je run_agent")
    run_button
    return (run_button,)


@app.cell(hide_code=True)
def demo_run(box_loop, result_loop, run_button):
    if result_loop.state != "correct":
        _out = mo.md("*Maak eerst `run_agent` goed in oefening 1: de demo gebruikt hem.*")
    elif not C.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif not run_button.value:
        _out = mo.md("*Druk op de knop om Qwen3 uit te voeren.*")
    else:
        _words = {"title": "Je run_agent met Qwen3-0.6B", "writing": "Qwen3 schrijft …", "loading": "Qwen3 laden …",
                  "alt": "De berichten van je run tot nu toe, één per rij, en wat Qwen3 nu schrijft."}
        _r = C.run_live(run_code(box_loop.value).names["run_agent"], lambda _html: mo.output.replace(mo.Html(_html)), _words, L.lang)
        if isinstance(_r, str):
            _out = mo.md({
                "agent": "*Je `run_agent` stopt met een fout op een bericht van Qwen3. Komt hij nog door de check?*",
                "load": "*Het laden van Qwen3 stopt met een fout. Sluit andere programma's en druk opnieuw op de knop; gebeurt het weer, meld het dan op GitHub.*",
                "memory": "*Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop.*",
            }[_r])
        else:
            _out = mo.Html(C.live_picture(_r, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_who_runs():
    mo.md(r"""
    ### Oefening 2. Wie voert de tool uit?
    """)
    return


@app.cell(hide_code=True)
def choice_who_runs_cell():
    choice_who_runs = L.choice(
        "who-runs",
        question="Qwen3 schrijft `<tool_call>{\"name\": \"pantry\", \"arguments\": {\"item\": \"milk\"}}</tool_call>`. Wie kijkt er in de voorraadkast?",
        options={
            "model": "Het model: de call schrijven voert de functie uit",
            "provider": "Hugging Face, waar het model vandaan kwam",
            "program": "Je programma: `run_agent` roept de functie `pantry` aan",
        },
        correct="program",
        why={
            "model": "Het model schrijft alleen tekst, ook als die tekst een tool call is. Iets moet die lezen en de functie uitvoeren. Kijk nog eens naar het plaatje van de loop.",
            "provider": "Hugging Face gaf je alleen het bestand met het model. Het draait nergens anders: op je computer kan het alleen tekst schrijven. Kijk nog eens naar het plaatje van de loop.",
            "program": "Het model schrijft alleen welke tool en welke argumenten. Je loop leest dat, voert `pantry` uit met `{\"item\": \"milk\"}`, en zet het resultaat in de berichten.",
        },
    )
    choice_who_runs
    return (choice_who_runs,)


@app.cell(hide_code=True)
def text_rl_agent():
    mo.md(r"""
    ### Oefening 3. Twee soorten agent
    """)
    return


@app.cell(hide_code=True)
def choice_rl_agent_cell():
    choice_rl_agent = L.choice(
        "rl-agent",
        question="Wat verschilt er tussen de RL agent van [Leren van rewards](/course/open/rl-what-is-rl?lang=nl) en je LLM agent?",
        options={
            "loop": "Alleen de RL agent werkt in een loop van handelen en zien wat er gebeurde",
            "choice": "Waar de keuzes vandaan komen: een policy geleerd uit rewards, of een language model dat de berichten leest",
            "goal": "Alleen de LLM agent heeft een doel",
        },
        correct="choice",
        why={
            "loop": "Je `run_agent` is ook een loop van handelen en zien: een tool call, dan zijn resultaat. Kijk nog eens naar de tabel van de twee agents.",
            "choice": "De RL agent leerde zijn policy door te proberen en rewards te krijgen. Je agent leerde niets terwijl hij werkte: zijn keuzes komen van Qwen3, dat de taak en elk resultaat tot nu toe leest.",
            "goal": "De RL agent heeft ook een doel: zoveel mogelijk return. Kijk nog eens naar de tabel van de twee agents.",
        },
    )
    choice_rl_agent
    return (choice_rl_agent,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Leren van rewards](/course/open/rl-what-is-rl?lang=nl) handelde een agent in een environment en zag hij wat er gebeurde; je LLM agent doet hetzelfde, met een language model dat kiest in plaats van een geleerde policy. In [Prompts die zichzelf controleren](/course/open/start-prompt-chains?lang=nl) zette je zelf prompts na elkaar; in een agent kiest het model de volgende stap. En in [Data als tekst: JSON](/course/open/py-json?lang=nl) las je JSON; elke tool call is JSON, die de `ask` van de course voor je loop leest. De lessen van het domein *Bouwen met AI* gaan verder: hoe je een tool beschrijft zodat een model hem goed gebruikt, wat een tool mag doen, en wanneer een vaste reeks stappen beter werkt dan een agent. In de volgende les, *Grafen: nodes, edges en paden*, blijkt een doolhof één voorbeeld van een graaf, en draait dezelfde search op elk netwerk.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_rl_agent, choice_who_runs, result_loop):
    L.wrap_up(result_loop, choice_who_runs, choice_rl_agent)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
