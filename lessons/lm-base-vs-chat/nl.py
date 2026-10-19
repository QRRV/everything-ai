# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les lm-base-vs-chat. Route: course/route/lessen/taal.yaml. Checks: checks.py.
# termen:allow instruct model, instruct models

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("lm-base-vs-chat", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_writes():
    quick_writes = L.choice(
        "quick-writes",
        question="Hoe schrijft een language model een tekst?",
        options={
            "lookup": "Het zoekt een heel antwoord op dat past bij de tekst tot nu toe",
            "right": "Het kiest een volgend token uit zijn kansen, voegt het toe en doet dat opnieuw",
            "once": "Het rekent alle tokens tegelijk uit en laat ze dan zien",
        },
        correct="right",
        why={
            "lookup": "Een language model geeft alleen kansen voor het volgende token; een tekst ontstaat door dat steeds opnieuw te doen. Zie [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl).",
            "once": "Het geeft de kansen van één volgend token tegelijk, en elk nieuw token hangt af van de tokens ervoor. Zie [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl).",
            "right": "Elk nieuw token komt bij de tekst tot nu toe, en het model geeft de kansen van het volgende.",
        },
    )
    quick_writes
    return (quick_writes,)


@app.cell(hide_code=True)
def quick_check_labels():
    quick_labels = L.choice(
        "quick-labels",
        question="Een language model leert met next-token prediction op heel veel tekst. Waar komen de labels vandaan?",
        options={
            "people": "Mensen schrijven bij elke zin het goede antwoord",
            "model": "Het model verzint ze terwijl het leert",
            "right": "Uit de tekst zelf, één label per volgend token",
        },
        correct="right",
        why={
            "people": "Niemand voegt labels toe: de tekst geeft zijn eigen labels. Zie [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl).",
            "model": "De labels liggen vast voordat de training begint: de tokens van de tekst. Zie [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl).",
            "right": "Het token dat echt volgt is het label, dus een model kan van elke tekst leren zonder dat iemand labels toevoegt.",
        },
    )
    quick_labels
    return (quick_labels,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je typt een vraag in een chatprogramma, en er komt een antwoord terug. Toch doet het model erachter alleen wat [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl) liet zien: het geeft kansen voor het volgende token. Een model dat alleen van tekst leerde, heeft na een vraag van alles zien komen: nog een vraag in een lijst met examenvragen, een reactie op een forum, de rest van een verhaal. Wat zorgt er dan voor dat het jou antwoordt?

    **Een voorbeeld uit het dagelijks leven.** Iemand die duizenden boeken heeft gelezen, weet hoe zinnen verdergaan. Zet die persoon voor het eerst achter een helpdesk, en dan moet die nog leren wat een goed antwoord aan een klant is: kort, ter zake en over de vraag. Dat leert die op de werkvloer, aan de hand van veel voorbeelden.
    """)
    return


@app.cell(hide_code=True)
def predict_turn():
    guess_turn = L.guess(
        "plays-turn",
        question="Het model van de course leerde van de toneelstukken van Shakespeare, en van niets anders. Het krijgt de tekst hieronder en schrijft door tot het een lege regel begint. Wat denk je dat het schrijft?\n\n```\nUSER:\nWhat is a token?\n\nASSISTANT:\n```",
        options={
            "right": "Een paar regels tekst in de stijl van de toneelstukken",
            "answer": "Een antwoord dat zegt dat een token een stukje tekst is",
            "nothing": "Niets, want tokens komen in de toneelstukken niet voor",
        },
        reveal=f"**Een paar regels tekst in de stijl van de toneelstukken.** Met seed 3 schrijft het:\n\n{C.quoted(C.write_turn(C.GUESS_TEXT, seed=3))}\n\nIn de toneelstukken komt na een naam met een dubbele punt op een eigen regel, zoals `ROMEO:`, wat die persoon zegt, tot een lege regel. Dus na `ASSISTANT:` schrijft het model een tekst alsof een personage dat Assistant heet aan het woord is: het neemt zijn beurt, maar het antwoordt niet. Het heeft nooit geleerd wat een goed antwoord is, alleen welke tekst er in de toneelstukken volgt. En het schrijft wel iets: een language model geeft altijd kansen voor een volgend token.",
    )
    guess_turn
    return (guess_turn,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **base model** is een language model dat alleen next-token prediction heeft geleerd, op heel veel tekst. Het schrijft elke tekst die het krijgt verder, met wat er in zijn teksten meestal volgt. Het model van de course is een base model, een heel klein. Het telt welk teken volgt op de laatste 7 tekens in de toneelstukken, zoals het model van [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl) dat deed met 3; komen die 7 nooit voor, dan kijkt het naar minder. Daardoor ziet het de vraag niet eens: met dezelfde seed schrijft het na `ASSISTANT:` hetzelfde, wat er ook gevraagd werd. Een echt base model ziet de hele tekst, en schrijft die toch alleen verder. Een groot base model dat op het web veel vragen met antwoorden heeft gezien, schrijft misschien iets wat op een antwoord lijkt, maar net zo goed nog een vraag, of de rest van een quiz.

        Een **chat model** is een base model dat verder getraind is op gesprekken: heel veel voorbeelden waarin iemand iets vraagt en een assistent goed antwoordt. Het doet nog steeds niets anders dan het volgende token voorspellen; alleen is na een vraag de waarschijnlijkste tekst nu een antwoord. Chatprogramma's gebruiken chat models. Modellen waarvan de naam eindigt op -Instruct of -Chat zijn chat models; sommige teksten noemen ze instruct models.

        Een gesprek bestaat uit berichten, en elk bericht heeft een **rol** die zegt van wie het komt: `system` voor instructies van wie het model inzet, zoals "Antwoord in het Nederlands."; `user` voor de persoon die typt; `assistant` voor het model zelf. In code is een gesprek een lijst berichten, elk een dictionary met de sleutels `"role"` en `"content"`: zo geven de meeste programma's een gesprek aan een chat model. Sommige programma's hebben een vierde rol, `tool`, voor wat een tool die het model gebruikte teruggaf.

        Toch krijgt het model één tekst. Het gesprek wordt uitgeschreven met een markering voor elk bericht, en het eindigt met de markering van `assistant`, zodat wat volgt het antwoord is. Het model schrijft tot het de markering geeft die zijn beurt afsluit. Chat models verschillen in hun markeringen, die ze in de training leerden; de les *Chat templates* laat ze zien. In deze les is de markering de rol in hoofdletters met een dubbele punt, op een eigen regel, zoals de toneelstukken de naam van een spreker schrijven, en sluit een lege regel een beurt af.

        Het model zelf onthoudt niets van de ene beurt tot de volgende. Voor elk nieuw antwoord gaat het hele gesprek er opnieuw in, als één tekst, met de eerdere antwoorden van het model erin als `assistant`-berichten.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **base model** | | een language model dat alleen heeft geleerd het volgende token te voorspellen, op heel veel tekst; het schrijft elke tekst die het krijgt verder | |
        | **chat model** | | een base model dat verder getraind is op gesprekken, zodat het als assistent antwoordt | |
        | **rol** | | van wie een bericht komt: `system`, `user` of `assistant` | `m["role"]` |
        | `system` | | instructies van wie het model inzet | `{"role": "system", "content": "Antwoord in het Nederlands."}` |
        | `user` | | de persoon die typt | `{"role": "user", "content": "Wie is Romeo?"}` |
        | `assistant` | | het model, ook in zijn eerdere antwoorden | `{"role": "assistant", "content": "Een jongeman uit Verona."}` |
        """),
        L.watch_out("`assistant` is een rol in de tekst, geen apart programma: hetzelfde model dat elk volgend token voorspelt, schrijft elk `assistant`-bericht. En een `system`-bericht is tekst zoals de rest, geen slot: een `user`-bericht kan het model nog steeds ergens anders heen leiden."),
        mo.md(r"""
        De figuur: het gesprek van het uitgewerkte voorbeeld hieronder, uitgeschreven als de ene tekst die het model krijgt. De beurt van het model zelf staat in kleur.
        """),
        mo.Html(C.turn_picture(C.EXAMPLE, C.write_turn(C.as_text_course(C.EXAMPLE), seed=6),
                               {"title": "Wat het model krijgt, en wat het schrijft", "system": "system: wie het model inzet",
                                "user": "user: de persoon", "assistant": "assistant: het model", "end": "lege regel: einde van de beurt",
                                "alt": "Het gesprek uitgeschreven als één tekst, regel voor regel: SYSTEM: en de instructie, een lege regel, USER: en de vraag, een lege regel, dan ASSISTANT: en de regel die het model schrijft, in kleur, tot een lege regel die zijn beurt afsluit. Rechts de rol van elk deel."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een gesprek aan een language model te geven:

    1. Schrijf het gesprek als een lijst berichten: elk een dictionary met de sleutels `"role"` en `"content"`.
    2. Schrijf elk bericht uit: zijn markering op een eigen regel, dan de inhoud, dan een lege regel.
    3. Eindig met de markering van `assistant`: wat volgt is het antwoord.
    4. Laat het model de tekst verder schrijven, tot de lege regel die zijn beurt afsluit.
    5. Voeg voor de volgende beurt het antwoord toe als `assistant`-bericht en de nieuwe vraag als `user`-bericht, en begin weer bij stap 2.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een gesprek met een `system`-bericht en één vraag, gegeven aan het base model van de course met seed 6. Het model kent alleen Engelse toneelstukken, dus het gesprek is in het Engels. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_turn():
    _MARKER = {"system": "SYSTEM:", "user": "USER:", "assistant": "ASSISTANT:"}
    # 1. het gesprek: een lijst berichten, elk met een rol en een inhoud
    _messages = [
        {"role": "system", "content": "Answer in one line."},
        {"role": "user", "content": "Who is Romeo?"},
    ]
    # 2. elk bericht uitgeschreven: zijn markering op een eigen regel, de inhoud, een lege regel
    _text = ""
    for _m in _messages:
        _text = _text + _MARKER[_m["role"]] + "\n" + _m["content"] + "\n\n"
    # 3. eindig met de markering van de assistant
    _text = _text + _MARKER["assistant"] + "\n"
    # 4. het base model van de course schrijft de tekst verder, tot de lege regel die zijn beurt afsluit
    _reply = C.write_turn(_text, seed=6)
    print(_text + _reply)
    # 5. de volgende beurt: het antwoord als assistant-bericht, de nieuwe vraag als user-bericht
    _messages.append({"role": "assistant", "content": _reply})
    _messages.append({"role": "user", "content": "And who is Juliet?"})
    print(len(_messages), _messages[2])
    return


@app.cell(hide_code=True)
def text_after_example():
    _reply = C.write_turn(C.as_text_course(C.EXAMPLE), seed=6)
    mo.md(f"""
    De tekst heeft drie delen: het `system`-bericht, het `user`-bericht, en de markering `ASSISTANT:` aan het eind. Het model schrijft "{_reply}" en stopt bij de lege regel. Dat is geen antwoord over Romeo: het model heeft nooit leren antwoorden, en met 7 tekens om op af te gaan ziet het de vraag niet eens. Een chat model zou van heel veel gesprekken hebben geleerd dat na een vraag over Romeo een antwoord over Romeo komt. Voor de volgende beurt heeft de lijst vier berichten: de regel van het model zelf staat erin als `assistant`-bericht, omdat het model zelf niets onthoudt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_chat():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Wat maakt een chat model?
    """)
    return


@app.cell(hide_code=True)
def choice_chat_cell():
    choice_chat = L.choice(
        "base-to-chat",
        question="Wat maakt van een base model een chat model?",
        options={
            "system": "Een `system`-bericht dat zegt dat het als assistent moet antwoorden",
            "markers": "De markeringen `USER:` en `ASSISTANT:` voor de berichten",
            "right": "Meer training, op heel veel gesprekken",
        },
        correct="right",
        why={
            "system": "Een `system`-bericht is tekst zoals de rest: een base model schrijft het alleen verder, zoals het model van de course de tekst van de gok na `ASSISTANT:` verder schreef.",
            "markers": "De tekst van de gok had die markeringen, en het base model van de course schreef toch een tekst uit een toneelstuk. Markeringen helpen alleen een model dat in de training leerde wat erna komt.",
            "right": "Training op heel veel gesprekken waarin een assistent goed antwoordt, maakt een antwoord de waarschijnlijke tekst na een vraag. De markeringen en de rollen zeggen alleen wie aan de beurt is.",
        },
    )
    choice_chat
    return (choice_chat,)


@app.cell(hide_code=True)
def text_exercise_role():
    mo.md(r"""
    ### Oefening 2. Welke rol?
    """)
    return


@app.cell(hide_code=True)
def choice_role_cell():
    choice_role = L.choice(
        "role-of",
        question="Een school zet een chat model op haar website, en geeft het model voor elk gesprek de instructie \"Beantwoord alleen vragen over het rooster.\" Welke rol heeft dat bericht?",
        options={
            "user": "`user`",
            "right": "`system`",
            "assistant": "`assistant`",
        },
        correct="right",
        why={
            "user": "`user` is de persoon die de vragen typt. De school zet het model in, en haar instructies zijn een `system`-bericht.",
            "assistant": "`assistant` is voor de eigen berichten van het model. De instructie komt van wie het model inzet.",
            "right": "De school zet het model in, en instructies van wie het model inzet zijn `system`-berichten.",
        },
    )
    choice_role
    return (choice_role,)


@app.cell(hide_code=True)
def text_exercise_as_text():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Het gesprek als één tekst

        `as_text(messages)` schrijft een gesprek uit als één tekst, zoals in stap 2 en 3 van het recept. De dictionary `MARKER`, de loop over de berichten en de laatste regel, met de markering van `assistant`, staan er al. Vervang de `...` door één regel voor stap 2. De check roept `as_text([{"role": "user", "content": "Hello"}])` aan, en dat moet `"USER:\nHello\n\nASSISTANT:\n"` geven; hij probeert ook andere gesprekken: met een `system`-bericht, met eerdere antwoorden, en een leeg gesprek.
        """),
        L.task("Laat `as_text` elk bericht aan `text` toevoegen: zijn markering op een eigen regel, de inhoud en een lege regel."),
    ])
    return


@app.cell(hide_code=True)
def exercise_as_text():
    box_as_text = L.code_box("as-text", lines=8)
    box_as_text
    return (box_as_text,)


@app.cell(hide_code=True)
def check_as_text(box_as_text):
    result_as_text = L.check("as-text", box_as_text.value)
    result_as_text
    return (result_as_text,)


@app.cell(hide_code=True)
def hints_as_text():
    L.hints(
        "as-text",
        [
            "Stap 2 van het recept, zoals in het uitgewerkte voorbeeld.",
            "De markering van een bericht staat in `MARKER`, onder zijn rol: zoek `m[\"role\"]` op. Na de markering komt `\"\\n\"`, na de inhoud `\"\\n\\n\"`: het einde van de regel en een lege regel.",
            "`text = text + MARKER[___] + \"\\n\" + m[___] + \"\\n\\n\"`.",
        ],
        why="Voor `[{\"role\": \"user\", \"content\": \"Hello\"}]` loopt de loop één keer: `text` wordt `\"USER:\\nHello\\n\\n\"`, en de laatste regel voegt `\"ASSISTANT:\\n\"` toe.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_next_turn():
    mo.vstack([
        mo.md(r"""
        ### Oefening 4. De volgende beurt

        `next_turn(messages, reply, question)` geeft het gesprek voor de volgende beurt, zoals in stap 5 van het recept: de berichten tot nu toe, dan het antwoord van het model, dan de nieuwe vraag. De kopie van de berichten tot nu toe staat er al: de toevoegingen komen in `new`, zodat de lijst die je kreeg blijft zoals hij was. Vervang de `...` door twee regels. De check roept `next_turn([{"role": "user", "content": "Hello"}], "Good day to you.", "Who are you?")` aan, en dat moet een lijst met drie berichten geven; hij probeert ook andere gesprekken.
        """),
        L.task("Laat `next_turn` het antwoord en de vraag aan `new` toevoegen, elk als bericht met de goede rol."),
    ])
    return


@app.cell(hide_code=True)
def exercise_next_turn():
    box_next_turn = L.code_box("next-turn", lines=7)
    box_next_turn
    return (box_next_turn,)


@app.cell(hide_code=True)
def check_next_turn(box_next_turn):
    result_next_turn = L.check("next-turn", box_next_turn.value)
    result_next_turn
    return (result_next_turn,)


@app.cell(hide_code=True)
def hints_next_turn():
    L.hints(
        "next-turn",
        [
            "Stap 5 van het recept, zoals in het uitgewerkte voorbeeld.",
            "Wie schreef het antwoord, en wie stelt de vraag? Elk wordt een dictionary met een `\"role\"` en een `\"content\"`, toegevoegd aan `new` met `append`.",
            "`new.append({\"role\": ___, \"content\": reply})`, en hetzelfde voor de vraag.",
        ],
        why="Het model schreef het antwoord, dus dat is een `assistant`-bericht; de vraag komt van de persoon, dus dat is een `user`-bericht. In die volgorde: het antwoord kwam voor de nieuwe vraag.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Een base model doet wat [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl) liet zien, en niets meer: het schrijft tekst verder. Een chat model doet hetzelfde, na meer training op gesprekken; de rollen en de markeringen maken van een gesprek één tekst die het verder kan schrijven. In de volgende les, *Waarom language models dingen verzinnen*, zie je wat het betekent dat ook een chat model waarschijnlijke tekst geeft, geen gecontroleerde feiten.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_chat, choice_role, result_as_text, result_next_turn):
    L.wrap_up(choice_chat, choice_role, result_as_text, result_next_turn)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
