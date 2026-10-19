# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les lm-hallucination. Route: course/route/lessen/taal.yaml. Checks: checks.py.
# termen:allow confabulatie

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("lm-hallucination", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Sampling: laat het model schrijven](/course/open/lm-sampling?lang=nl) en [Base model of chat model?](/course/open/lm-base-vs-chat?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rare():
    quick_rare = L.choice(
        "quick-rare",
        question="Je model schrijft met sampling. Na een bepaald teken heeft een zeldzaam teken een kans van 0,01. Wat gebeurt ermee?",
        options={
            "never": "Het wordt nooit getrokken: het is te onwaarschijnlijk",
            "right": "Het wordt af en toe getrokken, ongeveer één keer op de 100",
            "most": "Het wordt meestal getrokken, omdat sampling zeldzame tekens voortrekt",
        },
        correct="right",
        why={
            "never": "Sampling trekt met de kansen van de rij, dus 0,01 komt af en toe voor; alleen greedy decoding neemt het nooit. Zie [Sampling: laat het model schrijven](/course/open/lm-sampling?lang=nl).",
            "most": "Sampling neemt elk teken ongeveer zo vaak als zijn kans zegt, en niet vaker. Zie [Sampling: laat het model schrijven](/course/open/lm-sampling?lang=nl).",
            "right": "Sampling trekt met de kansen van de rij: 0,01 betekent ongeveer één trekking op de 100.",
        },
    )
    quick_rare
    return (quick_rare,)


@app.cell(hide_code=True)
def quick_check_chat():
    quick_chat = L.choice(
        "quick-chat",
        question="Een chat model krijgt een vraag. Wat doet het om die te beantwoorden?",
        options={
            "lookup": "Het zoekt het antwoord op in een voorraad gecontroleerde feiten",
            "copy": "Het kiest een van de antwoorden die het uit de training bewaart",
            "right": "Het schrijft de tekst verder, token voor token, met wat er na de vraag waarschijnlijk is",
        },
        correct="right",
        why={
            "lookup": "Een chat model heeft geen voorraad feiten om in te zoeken: het voorspelt het volgende token, zoals een base model. Zie [Base model of chat model?](/course/open/lm-base-vs-chat?lang=nl).",
            "copy": "Een chat model bewaart geen lijst met antwoorden: het schrijft zijn antwoord token voor token, met wat er waarschijnlijk volgt. Zie [Base model of chat model?](/course/open/lm-base-vs-chat?lang=nl).",
            "right": "Een chat model voorspelt nog steeds alleen het volgende token; na een vraag is de waarschijnlijke tekst een antwoord.",
        },
    )
    quick_chat
    return (quick_chat,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Vraag een chat model om een citaat van een bekend persoon, of om de bron van een bewering, en je krijgt er een: in de goede stijl, met een titel en een jaartal. Af en toe is het citaat nooit gezegd, en bestaat het boek niet. Het model loog niet, en er ging niets kapot: het deed wat elk language model doet, precies datgene wat het nuttig maakt.

    **Een voorbeeld uit het dagelijks leven.** Iemand heeft veel wijnrecensies gelezen, maar een bepaalde wijn nooit geproefd. Vraag ernaar, en het antwoord komt meteen, met de goede woorden: een vleugje eikenhout, een lange afdronk. Het klinkt als een kenner, en het is verzonnen uit wat zulke antwoorden meestal zeggen.
    """)
    return


@app.cell(hide_code=True)
def predict_romeo():
    _found = C.found()
    guess_romeo = L.guess(
        "romeo-lines",
        question=f"Het model van de course, dat van de toneelstukken leerde, schrijft de eerste regel van {C.N} teksten van Romeo, elk met een andere seed. Hoeveel van die {C.N} regels zegt Romeo volgens jou echt ergens in de toneelstukken?",
        options={
            "most": "De meeste: het heeft ze uit de toneelstukken geleerd",
            "half": "Ongeveer de helft",
            "right": "Bijna geen enkele",
        },
        reveal=f"**Bijna geen enkele: {len(_found)} van de {C.N}.** Alleen \"{_found[0]}\" is een echte regel uit de toneelstukken. De andere {C.N - len(_found)} klinken als Romeo, in de woorden van de toneelstukken, en niemand in de toneelstukken zegt ze; de eerste vijf zijn \"{C.made_up()[0]}\", \"{C.made_up()[1]}\", \"{C.made_up()[2]}\", \"{C.made_up()[3]}\" en \"{C.made_up()[4]}\". Het model weet niet welke van zijn regels echt zijn: het schrijft waarschijnlijke tekst, en een verzonnen regel in de stijl van de toneelstukken is ook waarschijnlijk.",
    )
    guess_romeo
    return (guess_romeo,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(f"""
        ## Het idee

        Een **hallucinatie** is output die vloeiend en zeker klinkt, maar onwaar of verzonnen is, zoals een citaat dat niemand ooit zei, of een bron die niet bestaat. Het komt voort uit wat een language model is: het geeft waarschijnlijke tekst. In de training leerde het welke tokens meestal op welke volgen; niets in next-token prediction controleert of een zin waar is. Vaak is een waarschijnlijke zin ook een ware, en daarom zijn language models nuttig. Maar een verzonnen citaat in de goede stijl is ook waarschijnlijk, token voor token.

        Hoe zeker het model klinkt, zegt je hier niets over. Het model van de course gaf elk teken van zijn eigen {C.N} regels van Romeo gemiddeld een kans van {L.number(C.mean_prob(C.made_up()), 2)}, en elk teken van de eerste {C.N} echte regels van Romeo {L.number(C.mean_prob(C.real()), 2)}, terwijl die echte regels in precies de tekst staan waarin het telde: het is net zo zeker van wat het verzon als van wat echt is. Een chat model klinkt op dezelfde manier zeker: het leerde van teksten waarin antwoorden vol overtuiging geschreven zijn.

        Een model kan ook de tekst die het kreeg verkeerd weergeven, zoals een samenvatting die een getal uit het artikel verandert: een waarschijnlijk getal, niet het getal dat er stond.

        Elk model heeft een **knowledge cutoff**: de datum tot waar zijn trainingstekst reikt. Over alles wat later is, weet het model niets, tenzij het in de tekst staat die het krijgt, bijvoorbeeld uit een zoekopdracht. Het schrijft er toch waarschijnlijke tekst over. De makers van een model zeggen meestal wanneer de cutoff is; het model zelf weet het vaak niet precies. De toneelstukken zijn meer dan 400 jaar oud, en het woord "computer" komt er niet in voor. Geef het model van de course "My computer" na `ROMEO:`, en het schrijft zonder aarzelen verder: "My computer{C.write_rest(C.START + "My computer", 1)}"

        Wat eruit komt, hangt ook af van details in de tekst die er niet toe zouden moeten doen. Geef het model van de course `ROMEO:` met een spatie na de dubbele punt, en met dezelfde seed 1 schrijft het niet meer "{C.romeo_line(1)}" maar "{C.write_rest(C.START.replace(":", ": "), 1).strip()}". Een groot model is veel minder gevoelig, maar andere woorden voor dezelfde vraag, of dezelfde voorbeelden in een andere volgorde, kunnen het antwoord nog steeds veranderen. Dat heet **prompt sensitivity**: de prompt, de tekst die je een model geeft, doet ertoe op manieren die niet zouden moeten. Eén antwoord, hoe zeker het ook klinkt, is dus nog niet nagekeken.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **hallucinatie** | | output die vloeiend en zeker klinkt, maar onwaar of verzonnen is | |
        | **knowledge cutoff** | | de datum tot waar de trainingstekst van een model reikt; over latere gebeurtenissen weet het model niets, tenzij het die krijgt | |
        | **prompt sensitivity** | | hoe sterk het antwoord van een model verandert door kleine aanpassingen in de tekst die het krijgt, de prompt, die er niet toe zouden moeten doen | |
        | `line in lines` | | `True` als `line` een van de items van `lines` is, zoals `in` de sleutels van een dictionary nakijkt | `"Whither?" in ["Whither?", "Away!"]` geeft `True` |
        """),
        L.watch_out("*Hallucinatie* is een geleend woord: in de geneeskunde betekent het zien of horen wat er niet is. Een language model ziet niets; het schrijft waarschijnlijke tekst, en soms is die tekst onwaar. Sommige onderzoekers zeggen liever confabulatie: een gat vullen met iets wat goed klinkt, zonder te weten dat het verzonnen is."),
        mo.md(f"""
        De figuur: hoe zeker het model is van elk van zijn {C.N} regels van Romeo, en van elk van de eerste {C.N} echte regels van Romeo. Elk rondje is een regel; het korte rechtopstaande streepje is het gemiddelde.
        """),
        mo.Html(C.sure_strips({"title": "Hoe zeker is het model?", "made": f"de eigen regels van het model ({len(C.found())} van {C.N} echt)",
                               "real": "echte regels van Romeo", "mean": "gemiddeld {m}", "axis": "gemiddelde kans per teken",
                               "number": lambda v: L.number(v, 2), "tick": lambda v: L.number(v, 1),
                               "alt": f"Twee rijen van {C.N} rondjes op één as van 0,4 tot 0,9: de gemiddelde kans per teken van de eigen regels van het model, en van echte regels van Romeo. De twee rijen liggen over elkaar, allebei vooral tussen 0,55 en 0,8, met hun gemiddelden dicht bij elkaar."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om na te gaan wat een model schrijft:

    1. Laat het model schrijven: hier de eerste regel van een tekst van Romeo, met verschillende seeds.
    2. Zoek elke regel op in een bron die je vertrouwt, hier de regels van de toneelstukken, met `in`.
    3. Tel de regels die je vond, en deel door het aantal regels: het deel dat klopt.
    4. Zet de kansen van het model ernaast: hoe zeker het was, zegt niets over wat je vond.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Vijf regels van Romeo van het model van de course, met seed 1 tot en met 5. `C.play_lines()` geeft elke regel van de toneelstukken, en `C.how_sure(line)` de gemiddelde kans die het model aan elk teken van een regel geeft, afgerond op 2 decimalen. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_check():
    # 1. het model schrijft de eerste regel van Romeo, met seed 1 tot en met 5
    _lines = []
    for _seed in range(1, 6):
        _lines.append(C.romeo_line(_seed))
    print(_lines)
    # 2. zoek elke regel op in de regels van de toneelstukken
    _plays = C.play_lines()
    _found = 0
    for _line in _lines:
        if _line in _plays:
            _found = _found + 1
    # 3. het deel van de regels dat echt in de toneelstukken staat
    print(_found / len(_lines))
    # 4. hoe zeker het model van elke regel was
    for _line in _lines:
        print(C.how_sure(_line), _line)
    return


@app.cell(hide_code=True)
def text_after_example():
    _sure = [C.how_sure(C.romeo_line(_s)) for _s in range(1, 6)]
    mo.md(f"""
    Geen van de vijf regels staat in de toneelstukken: het deel is 0. Toch was het model er zeker van, met gemiddeld tussen {L.number(min(_sure), 2)} en {L.number(max(_sure), 2)} per teken, minstens zo zeker als van de echte regels in de figuur. Alleen de bron vertelt je welke regels echt zijn; de kansen kunnen dat niet.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_cutoff():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Na de cutoff
    """)
    return


@app.cell(hide_code=True)
def choice_cutoff_cell():
    choice_cutoff = L.choice(
        "cutoff",
        question="De trainingstekst van een model reikt tot januari 2025. Iemand vraagt het naar een gebeurtenis van juni 2025, en geeft het geen tekst over die gebeurtenis. Wat kan er gebeuren?",
        options={
            "right": "Het schrijft een vloeiend antwoord dat best verzonnen kan zijn",
            "always": "Het zegt altijd dat het niets over die gebeurtenis kan weten",
            "search": "Het zoekt de gebeurtenis zelf op internet op",
        },
        correct="right",
        why={
            "always": "Een model kan getraind zijn om te zeggen dat het iets niet weet, maar dat doet het niet altijd: het schrijft waarschijnlijke tekst, en een vloeiend antwoord is ook waarschijnlijk.",
            "search": "Een model krijgt alleen de tekst die het gegeven wordt. Sommige chatprogramma's zoeken op internet en voegen wat ze vinden aan die tekst toe, maar het model zelf zoekt niets op.",
            "right": "Over juni 2025 weet het model niets, maar het geeft nog steeds kansen voor elk volgend token, dus er kan een vloeiend antwoord uitkomen, verzonnen uit wat zulke antwoorden meestal zeggen.",
        },
    )
    choice_cutoff
    return (choice_cutoff,)


@app.cell(hide_code=True)
def text_exercise_found():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Nakijken in een bron

        `found_share(lines, source)` geeft het deel van `lines` dat echt in `source` staat, zoals in stap 2 en 3 van het recept. De loop en de laatste regel staan er al. Vervang de `...` door een `if` die een regel telt als die in `source` staat. De check roept `found_share(["Good night.", "Be merry.", "Whither?", "Away, away!", "Away!"], ["Whither?", "Good night.", "Away!"])` aan, en dat moet `0.6` geven; hij probeert ook andere regels en bronnen, ook een lege bron.
        """),
        L.task("Laat `found_share` elke regel tellen die in `source` staat."),
    ])
    return


@app.cell(hide_code=True)
def exercise_found():
    box_found = L.code_box("found-share", lines=7)
    box_found
    return (box_found,)


@app.cell(hide_code=True)
def check_found(box_found):
    result_found = L.check("found-share", box_found.value)
    result_found
    return (result_found,)


@app.cell(hide_code=True)
def hints_found():
    L.hints(
        "found-share",
        [
            "Stap 2 en 3 van het recept, zoals in het uitgewerkte voorbeeld.",
            "Voor elke regel: is het een van de regels van `source`? `line in source` geeft `True` of `False`. Zo ja, dan gaat `found` één omhoog.",
            "`if line ___ source:`, en daarin `found = found + ___`.",
        ],
        why="Van de vijf regels staan \"Good night.\", \"Whither?\" en \"Away!\" in de bron; \"Away, away!\" niet, want de bron heeft alleen \"Away!\" zelf. 3 van de 5 is 0,6.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_sure():
    mo.md(r"""
    ### Oefening 3. Zeker, en toch verzonnen
    """)
    return


@app.cell(hide_code=True)
def choice_sure_cell():
    _line = C.romeo_line(30)
    choice_sure = L.choice(
        "sure-not-true",
        question=f"Het model gaf de tekens van zijn regel \"{_line}\" gemiddeld een kans van {L.number(C.how_sure(_line), 2)}, hoger dan bij elk van de eerste {C.N} echte regels van Romeo. Toch zegt niemand die regel in de toneelstukken. Wat laat dat zien?",
        options={
            "real": "De regel moet ergens echt zijn, want het model is er zo zeker van",
            "right": "Een kans zegt hoe waarschijnlijk een regel is, niet of hij waar is",
            "broken": "Het model is kapot: een goed model is alleen zeker van echte regels",
        },
        correct="right",
        why={
            "real": "De toneelstukken zijn alles waar het model van leerde, en de regel staat er niet in: het model is zeker van een regel die het verzon.",
            "broken": "Dit is geen fout van dit ene model: elk language model geeft waarschijnlijke tekst, en een verzonnen regel kan net zo waarschijnlijk zijn als een echte.",
            "right": "Het model is zeker omdat de regel teken voor teken op de toneelstukken lijkt. Of Romeo hem ooit zei, kan het model niet zien; daarvoor kijk je in de bron.",
        },
    )
    choice_sure
    return (choice_sure,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Een language model geeft waarschijnlijke tekst, zoals in [Sampling: laat het model schrijven](/course/open/lm-sampling?lang=nl) en [Base model of chat model?](/course/open/lm-base-vs-chat?lang=nl); waarschijnlijk is vaak waar, maar niet altijd, en hoe zeker het model is, zegt niet welke van de twee. Kijk wat ertoe doet dus na in een bron die je vertrouwt. Met *Toon verdieping* aan komen er eerst twee: *Encoders, decoders en de GPT-lijn*, over de soorten language models, en *Kijk in de kansen van een echte LLM voor het volgende token*, waarin je een klein open model laadt en zijn kansen leest. In de volgende les van de hoofdroute, *Wat een paper, een preprint en peer review zijn*, ga je zelf onderzoek lezen, waar ook elke bewering een bron nodig heeft.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_cutoff, result_found, choice_sure):
    L.wrap_up(choice_cutoff, result_found, choice_sure)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
