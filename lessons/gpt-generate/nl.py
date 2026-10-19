# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson gpt-generate. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("gpt-generate", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Sampling: laat het model schrijven](/course/open/lm-sampling?lang=nl) en [Temperature: gedurfder of veiliger tekst](/course/open/lm-temperature?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sees():
    quick_sees = L.choice(
        "quick-sees",
        question="Je bigram-model schrijft \"thithe thes s hand\": veel korte stukjes, weinig hele woorden. Waarom maakt het zelden een woord af?",
        options={
            "rows": "Zijn tabel heeft te weinig rijen: één per teken",
            "right": "Het ziet alleen het laatste teken van de tekst",
            "hot": "Zijn temperature is te hoog",
        },
        correct="right",
        why={
            "rows": "Een rij per teken is wat een bigram-model is: de rij van het laatste teken is alles wat het gebruikt. Zie [Sampling: laat het model schrijven](/course/open/lm-sampling?lang=nl).",
            "hot": "Een lagere temperature maakt het veiliger, maar het ziet nog steeds maar één teken: het kan niet weten in welk woord het zit. Zie [Sampling: laat het model schrijven](/course/open/lm-sampling?lang=nl).",
            "right": "Vandaag schrijft je GPT met dezelfde loop, en ziet hij tot `block` tekens terug.",
        },
    )
    quick_sees
    return (quick_sees,)


@app.cell(hide_code=True)
def quick_check_cold():
    quick_cold = L.choice(
        "quick-cold",
        question="Bij welke temperature neemt een model bijna altijd zijn waarschijnlijkste volgende teken, zoals greedy decoding?",
        options={
            "one": "1: de kansen zoals het model ze geeft",
            "big": "Een grote, zoals 10",
            "right": "Een die dicht bij 0 ligt, zoals 0,1",
        },
        correct="right",
        why={
            "one": "Bij 1 blijven de kansen zoals ze zijn, en worden ook minder waarschijnlijke tekens getrokken. Zie [Temperature: gedurfder of veiliger tekst](/course/open/lm-temperature?lang=nl).",
            "big": "Een grote temperature maakt de kansen vlakker, bijna gelijk voor elk teken. Zie [Temperature: gedurfder of veiliger tekst](/course/open/lm-temperature?lang=nl).",
            "right": "Delen door 0,1 maakt de verschillen tussen de logits tien keer zo groot, dus wint de grootste bijna altijd.",
        },
    )
    quick_cold
    return (quick_cold,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        In [Sampling: laat het model schrijven](/course/open/lm-sampling?lang=nl) schreef je bigram-model met een loop: de kansen op het volgende teken, een teken dat ermee getrokken wordt, achter de tekst gezet, en opnieuw. Je GPT is nu getraind, in [Project: je GPT op een boek trainen](/course/open/gpt-train-chars?lang=nl), en geeft ook kansen op het volgende teken, voor een hele tekst in plaats van één teken. In deze projectstap schrijft je loop met je GPT, met één regel anders. Zo schrijft elke GPT, ook de grote: één token per keer.
        """),
        draw.project_steps(["Letters en paren tellen", "Een bigram-model door te tellen", "Hetzelfde model, geleerd door een neural network",
                            "Je eigen tokenizer", "Je eigen GPT", "Een klein LLM pretrainen", "Het LLM instructies en voorkeuren laten volgen"],
                           4, here_label="← vandaag", title="Waar je project staat: je eigen LLM"),
    ])
    return


@app.cell(hide_code=True)
def predict_text():
    guess_text = L.guess(
        "gpt-text",
        question="Je GPT schrijft 300 tekens na \"ROMEO:\" en een nieuwe regel, bij een temperature van 0,8. Hoe ziet de tekst eruit, denk je?",
        options={
            "soup": "Zoals bij het bigram-model: korte stukjes, hier en daar een woord",
            "right": "Vooral echte woorden, namen en regels van een toneelstuk, in zinnen die weinig betekenen",
            "copy": "Hele stukken tekst uit de toneelstukken, woord voor woord",
        },
        reveal="**Vooral echte woorden, namen en regels van een toneelstuk.** Met een validation loss van 1,7 of minder, ver onder het bigram-model, heeft je GPT geleerd hoe woorden gespeld worden, dat een naam in hoofdletters met een dubbele punt een tekst van een personage begint, en hoe lang een versregel is. Wat de woorden betekenen, heeft hij niet geleerd: een miljoen of tien miljoen parameters, getraind op één boek, is daar veel te weinig voor. En hij kopieert geen hele stukken: hij trekt elk teken opnieuw, en één minder waarschijnlijke trekking stuurt de tekst een eigen kant op.",
    )
    guess_text
    return (guess_text,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Schrijven met een GPT is de loop van je bigram-model; alleen de kansen komen ergens anders vandaan:

        - **het venster:** je GPT leest hooguit `block` tekens, dus hij krijgt de laatste `block` tekens van de tekst; een kortere tekst leest hij helemaal;
        - **de laatste plek:** hij geeft logits op elke plek van het venster, elk over het teken na die plek. Alleen die van de laatste plek gaan over het volgende teken van de tekst;
        - **de temperature:** zoals in [Temperature: gedurfder of veiliger tekst](/course/open/lm-temperature?lang=nl), deel die logits door $\tau$ vóór softmax;
        - **de loop:** trek een teken met de kansen, zet het achter de tekst, en begin opnieuw: één forward pass voor elk nieuw teken.

        Om te schrijven heeft je GPT geen gradients nodig: met `torch.no_grad()` bouwt PyTorch geen graaf, en dat spaart geheugen en tijd.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent |
        |---|---|
        | **venster** | de laatste `block` tekens van de tekst: wat de GPT leest om het volgende te geven |
        | `num_embeddings` | het aantal rijen van een tabel, hier `model.pos`, de tabel van plekken van je GPT: zijn `block` |
        """),
        L.watch_out("Elk nieuw teken kost een hele forward pass over het venster, ook voor alle tekens die de GPT in de stap ervoor al las. Voor 300 nieuwe tekens zijn dat 300 passes. Grote modellen bewaren de keys en values van eerdere plekken in een KV cache, zodat een stap alleen nog het nieuwe teken hoeft te verwerken; een latere les bouwt er een."),
        mo.md(r"""
        De figuur: één stap. Het venster bestaat uit de laatste tekens van de tekst; je GPT geeft logits op elke plek, en alleen die van de laatste plek, gedeeld door $\tau$ en door softmax, geven de kansen op het volgende teken. Een teken dat ermee getrokken wordt, gaat achter de tekst, en de volgende stap begint.
        """),
        mo.Html(C.loop_picture({"title": "Eén stap van het schrijven", "window": "venster", "gpt": "je GPT", "logits": "logits", "probs": "÷ τ, softmax",
                                "draw": "trekken, erbij, opnieuw",
                                "alt": "De tekst ROMEO, een dubbele punt, een nieuwe regel en No, in vakjes, met een nieuw vakje aan het eind voor het volgende teken. De laatste zes tekens zijn het venster, dat je GPT in gaat. Daaronder een rij logits, één voor elke plek van het venster; alleen de laatste is gekleurd. Daaruit, na delen door tau en softmax, drie balken: een komma 0,39, een t 0,23 en een spatie 0,16, de GPT van de course na deze tekst bij een temperature van 0,8. Een pijl gaat van de balken terug naar het nieuwe vakje: trekken, erbij, opnieuw."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Schrijven met een GPT:

    1. De laatste `block` tekens van de tekst, als ids.
    2. De logits van de GPT daarvoor, zonder graaf; neem die van de laatste plek.
    3. Deel ze door $\tau$ en neem softmax: de kansen op het volgende teken.
    4. Trek een teken met die kansen, en zet het achter de tekst.
    5. Ga terug naar stap 1 tot de tekst lang genoeg is.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 2 tot en met 4 voor de logits die een GPT gaf voor een tekst van 3 tekens, met een vocabulary van 3: `"a"`, `"e"` en `"t"`. De logits hebben shape (1, 3, 3): één tekst, 3 plekken, 3 scores op elke plek. Stap 2 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_step():
    import numpy as np
    import torch

    _chars = ["a", "e", "t"]
    _logits = torch.tensor([[[2.0, 0.5, 0.1], [0.3, 1.8, 0.2], [0.4, 0.2, 2.2]]])
    # 2. de logits over het volgende teken
    _last = _logits[0, 0]  # maak af: _logits[0, -1], die van de laatste plek
    # 3. een temperature van 0,8, en softmax
    _p = torch.softmax(_last / 0.8, dim=0).numpy()
    print(np.round(_p, 3))
    # 4. trek het volgende teken
    _rng = np.random.default_rng(1)
    print(_chars[_rng.choice(3, p=_p)])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Zoals hij is, neemt de cel de logits van de eerste plek: `[0.802 0.123 0.075]`, en trekt hij `a`. Die gaan over het teken na het eerste, dat de tekst al heeft. Afgemaakt, met de laatste plek: `[0.089 0.069 0.842]`, en hij trekt `t`, met dezelfde seed.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_passes():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Hoeveel passes?
    """)
    return


@app.cell(hide_code=True)
def choice_passes_cell():
    choice_passes = L.choice(
        "passes",
        question="Je GPT schrijft een tekst van 300 tekens, na een begin van 7: `\"ROMEO:\"` en een nieuwe regel. Hoeveel forward passes doet hij?",
        options={
            "one": "1: één pass geeft logits op elke plek",
            "all": "300: één voor elk teken van de tekst",
            "right": "293: één voor elk nieuw teken",
        },
        correct="right",
        why={
            "one": "Eén pass geeft logits op elke plek, maar alleen de laatste plek gaat over een teken dat de tekst nog niet heeft. Het volgende heeft een pass nodig over de tekst met dat teken erin.",
            "all": "De 7 tekens van het begin zijn gegeven: daarvoor hoeft niets getrokken te worden.",
            "right": "Elk ervan over een venster van hooguit `block` tekens. Een KV cache, in een latere les, bespaart het meeste van dat werk zolang de tekst in het venster past.",
        },
    )
    choice_passes
    return (choice_passes,)


@app.cell(hide_code=True)
def text_exercise_probs():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De kansen op het volgende teken

        Maak `gpt_probs(model, row_of, text, tau)` af: stap 1 tot en met 3 van het recept. `row_of` geeft het id van elk teken, zoals in je `generate` uit [Sampling: laat het model schrijven](/course/open/lm-sampling?lang=nl); de functie geeft een NumPy-array terug, zoals een rij van je bigram-tabel was. De check maakt een kleine GPT van jou na `torch.manual_seed(0)`, voor de 5 tekens `"abcde"` en teksten van hooguit 4, en roept `gpt_probs(model, row_of, "abc", 1.0)` aan; daarna probeert hij een langere tekst, twee temperatures en een GPT die 6 tekens leest. Als hij goed is, bewaart de check hem in je project, als `work/projects/llm/gpt_probs.py`.
        """),
        L.task("Laat `gpt_probs` de kansen op het volgende teken geven, voor elke tekst en temperature."),
    ])
    return


@app.cell(hide_code=True)
def exercise_probs():
    box_probs = L.code_box("gpt-probs", lines=13)
    box_probs
    return (box_probs,)


@app.cell(hide_code=True)
def check_probs(box_probs):
    result_probs = L.check("gpt-probs", box_probs.value)
    result_probs
    return (result_probs,)


@app.cell(hide_code=True)
def hints_probs():
    L.hints(
        "gpt-probs",
        [
            "De ids: een loop over de laatste `block` tekens van `text`, met `row_of`. Dan je GPT op een tensor van één tekst, `torch.tensor([ids])`, binnen `with torch.no_grad():`.",
            "De logits hebben in de check shape (1, n, 5): één tekst, n plekken, 5 scores op elke plek. Stap 2 van het afgemaakte uitgewerkte voorbeeld neemt de laatste plek; stap 3 deelt door de temperature en neemt softmax, `dim=0`, als NumPy.",
        ],
        why="Alleen de logits van de laatste plek gaan over een teken dat de tekst nog niet heeft; de andere gaan over tekens die er al staan.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_generate(result_probs):
    result_probs  # de melding verandert als oefening 2 je gpt_probs bewaart
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Je loop, met je GPT

        Maak `gpt_generate(model, chars, start, n, tau, seed)` af: stap 4 en 5 van het recept, de loop van je `generate` uit [Sampling: laat het model schrijven](/course/open/lm-sampling?lang=nl) met de kansen van je `gpt_probs` in plaats van een rij van de tabel. De eerste regels zijn die van je `generate`, en importeren je `gpt_probs` uit je project. De check roept `gpt_generate(model, chars, "ab", 9, 1.0, 1)` aan met de kleine GPT van oefening 2, en probeert andere beginteksten, seeds en temperatures, en een GPT die 6 tekens leest. Als hij goed is, bewaart de check hem in je project, als `work/projects/llm/gpt_generate.py`.
        """),
        L.course_versions(),
        L.task("Laat `gpt_generate` schrijven met je GPT, één getrokken teken per keer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_generate():
    box_generate = L.code_box("gpt-generate", lines=14)
    box_generate
    return (box_generate,)


@app.cell(hide_code=True)
def check_generate(box_generate):
    result_generate = L.check("gpt-generate", box_generate.value)
    result_generate
    return (result_generate,)


@app.cell(hide_code=True)
def hints_generate():
    L.hints(
        "gpt-generate",
        [
            "De loop van je `generate` uit lm-sampling: zolang de tekst korter is dan `n`, de kansen op het volgende teken, een getrokken teken, achter de tekst gezet.",
            "Alleen de regel van de kansen verandert: in plaats van de rij van het laatste teken, je `gpt_probs` voor de hele tekst tot nu toe.",
        ],
        why="Je GPT leest de hele tekst tot nu toe, tot `block` tekens; het bigram-model las alleen het laatste teken. Die ene regel is het hele verschil tussen de twee.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    De cel hieronder laat je GPT 300 tekens schrijven na "ROMEO:" en een nieuwe regel, met je `gpt_generate`: de GPT die je trainde in [Project: je GPT op een boek trainen](/course/open/gpt-train-chars?lang=nl), of de GPT van de course als je er nog geen getraind hebt. Daaronder schrijft je bigram-model na hetzelfde begin, met je `generate` uit lm-sampling. Kies een temperature.
    """)
    return


@app.cell(hide_code=True)
def temperature_cell():
    temperature = mo.ui.dropdown(options={C.number(t, 1, L.lang): t for t in C.TEMPERATURES}, value=C.number(0.8, 1, L.lang), label="Temperature")
    temperature
    return (temperature,)


@app.cell(hide_code=True)
def demo_write(temperature, result_probs, result_generate):
    result_probs, result_generate  # opnieuw draaien als oefening 2 of 3 je module bewaart
    _result = C.demo(temperature.value)
    if _result is None:
        _out = mo.md("*Maak eerst oefening 2 en 3 af: dan staan je `gpt_probs` en je `gpt_generate` in je project, en gebruikt deze cel ze.*")
    elif isinstance(_result, str):
        _out = mo.md({
            "slow": "*Schrijven duurt hier te lang: meer dan een minuut. Doet je `gpt_probs` of je `gpt_generate` meer werk dan één forward pass voor elk nieuw teken?*",
            "gpt": "*Je GPT, of je code in `work/projects/llm/`, stopt met een fout tijdens het schrijven. Halen je modules hun checks nog? Verwijder een bestand dat je veranderd hebt, en check zijn oefening opnieuw.*",
        }[_result])
    else:
        _who = "**Je GPT**, getraind in [Project: je GPT op een boek trainen](/course/open/gpt-train-chars?lang=nl)" if _result["source"] == "yours" else "**De GPT van de course**, omdat je er nog geen getraind hebt"
        _stand = "" if not _result["stand_ins"] else " De versie van de course sprong in voor " + ", ".join(f"`{_m}`" for _m in _result["stand_ins"]) + ", zoals de melding bij oefening 3 zegt."
        _out = mo.md(f"""{_who} (validation loss {C.number(_result['best'], 2, L.lang)}), bij een temperature van {C.number(temperature.value, 1, L.lang)}:

{C.shown_text(_result['gpt'])}

**Je bigram-model**, bij een temperature van 1:

{C.shown_text(_result['bigram'])}

Dezelfde loop, één regel anders: de GPT leest de hele tekst tot nu toe, tot `block` tekens; het bigram-model las alleen het laatste teken.{_stand}""")
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De sampling-loop uit [Sampling: laat het model schrijven](/course/open/lm-sampling?lang=nl) en de temperature uit [Temperature: gedurfder of veiliger tekst](/course/open/lm-temperature?lang=nl), nu met de kansen van je GPT uit [Project: je GPT op een boek trainen](/course/open/gpt-train-chars?lang=nl). Je project `llm` heeft nu `gpt_probs.py` en `gpt_generate.py`: je eigen GPT, van tekst naar tekst. Hierna: *Project: je GPT op je eigen tokens*, waar je GPT traint op de tokens van je eigen BPE-tokenizer in plaats van op tekens.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_passes, result_probs, result_generate):
    L.wrap_up(choice_passes, result_probs, result_generate)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
