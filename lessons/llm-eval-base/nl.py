# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-eval-base. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import math

    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-eval-base", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Perplexity](/course/open/lm-perplexity?lang=nl) en [Base model of chat model?](/course/open/lm-base-vs-chat?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_ppl():
    quick_ppl = L.choice(
        "quick-ppl",
        question="Op een tekst die het nooit zag, heeft een model een gemiddelde loss van 2,0 nats per token. Ongeveer hoe groot is zijn perplexity?",
        options={
            "hundred": "100: 10 tot de macht 2",
            "two": "2: de loss zelf",
            "right": "7,4: e tot de macht 2",
        },
        correct="right",
        why={
            "two": "De loss is de log van de perplexity, in nats: de perplexity is $e^{2{,}0}$. Zie [Perplexity](/course/open/lm-perplexity?lang=nl).",
            "hundred": "Een loss in nats gebruikt de natuurlijke log, dus de perplexity is $e^{2{,}0}$, niet $10^{2{,}0}$. Zie [Perplexity](/course/open/lm-perplexity?lang=nl).",
            "right": "$e^{2{,}0} \\approx 7{,}4$: alsof het model elke keer uit ongeveer 7 tokens kiest. Vandaag zie je waarom dat getal van de tokens afhangt.",
        },
    )
    quick_ppl
    return (quick_ppl,)


@app.cell(hide_code=True)
def quick_check_base():
    quick_base = L.choice(
        "quick-base",
        question="Je vraagt je model uit de capstone, een base model, \"What is two plus two?\", en het schrijft door met een verhaal. Wat zegt dat over hoe goed het zijn verhalen leerde?",
        options={
            "right": "Weinig: een base model leerde alleen tekst voortzetten",
            "badly": "Dat het ze slecht leerde: een goed model antwoordt 4",
            "more": "Dat het meer training op dezelfde verhalen nodig heeft",
        },
        correct="right",
        why={
            "badly": "Antwoorden is waarvoor een chat model getraind wordt; een base model zet de tekst voort, hoe goed het ook leerde. Zie [Base model of chat model?](/course/open/lm-base-vs-chat?lang=nl).",
            "more": "Meer van dezelfde verhalen maakt het beter in verhalen, niet in antwoorden: dat leert een chat model door training op gesprekken. Zie [Base model of chat model?](/course/open/lm-base-vs-chat?lang=nl).",
            "right": "Dus test je een base model niet met vragen, maar met hoe goed het tekst voorspelt: zijn loss op tekst die het nooit zag.",
        },
    )
    quick_base
    return (quick_base,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je pretrainde een model. Is het goed? Een getal op zich zegt weinig: je wilt het naast andere modellen leggen, zoals GPT-2. Maar elk model knipt tekst in zijn eigen tokens, en een loss of een perplexity per token hangt af van hoe groot die tokens zijn. Daarom kun je de loss van een base model ook per byte tekst geven: The Pile, een grote open dataset met tekst, meet in bits per byte hoe goed modellen hem voorspellen, omdat dat niet afhangt van hoe een tokenizer de tekst knipt (Gao en anderen, 2020).

    **Een voorbeeld uit het dagelijks leven.** Twee hardlopers zeggen elk hoeveel stappen ze nodig hadden voor een wedstrijd. De een zet kleine stappen, de ander grote: alleen de tijd op hetzelfde parcours zegt wie sneller is.
    """)
    return


@app.cell(hide_code=True)
def predict_gpt2():
    _m = C.MEASURED
    guess_gpt2 = L.guess(
        "gpt2",
        question=f"In de run van de course uit de capstone heeft een model van {C.number(_m['yours']['params'], 0, L.lang)} parameters, getraind op TinyStories, een perplexity van {C.number(_m['yours']['ppl'], 1, L.lang)} per token op {C.STORIES} verhalen die het nooit zag. GPT-2 small heeft {C.number(_m['gpt2']['params'] / 1e6, 0, L.lang)} miljoen parameters en werd getraind op webpagina's. Wat is zijn perplexity per token op dezelfde verhalen, met zijn eigen tokens?",
        options={
            "lower": "Lager: een model van die grootte voorspelt beter",
            "right": "Hoger",
            "same": "Ongeveer even hoog",
        },
        reveal=f"**Hoger: {C.number(_m['gpt2']['ppl'], 1, L.lang)}.** Per token lijkt het kleine model veel beter. Maar de tokens van GPT-2 zijn groter: het heeft {C.number(_m['gpt2']['tokens'], 0, L.lang)} tokens nodig voor deze verhalen, het kleine model {C.number(_m['yours']['tokens'], 0, L.lang)}. Elke gok van GPT-2 dekt meer tekst, dus elke gok is moeilijker. Welk model de verhalen beter voorspelt, meet deze les.",
    )
    guess_gpt2
    return (guess_gpt2,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Scoor een base model op tekst die het nooit zag.** Knip de tekst in de eigen tokens van het model, en laat het in vensters zo lang als die waarop het trainde elk token voorspellen uit de tokens ervoor in zijn venster. De loss van elk token, in nats, zegt hoe verrast het model was.

    **Een getal per token hangt af van de tokens.** Een tokenizer met grote tokens heeft er weinig nodig, en elke voorspelling dekt meer tekst: zijn loss per token is hoger, ook voor een model dat de tekst even goed voorspelt. Een perplexity per token is alleen te vergelijken tussen modellen met dezelfde tokenizer.
    """)
    return


@app.cell(hide_code=True)
def picture_cut():
    mo.vstack([
        mo.md("Het beeld: een zin uit de verhalen die deze les scoort, geknipt in de tokens van het model dat de eerste knop scoort, en in die van GPT-2. De bytes zijn dezelfde; GPT-2 heeft er minder tokens voor nodig, dus elke gok ervan dekt meer tekst."),
        C.cut_picture({"title": "Dezelfde {n} bytes, andere tokens", "ours": "het model van de eerste knop: {n} tokens", "gpt2": "GPT-2: {n} tokens",
                       "alt": "Een zin van {n} bytes, geknipt in de tokens van twee tokenizers: GPT-2 heeft er minder nodig."}),
    ])
    return


@app.cell(hide_code=True)
def text_idea_bpb():
    mo.md(r"""
    **Bits per byte.** Tel de losses van alle tokens op, de verrassing van de hele tekst. Maak van nats bits, door te delen door $\ln 2$, en verdeel ze over de bytes van de tekst in UTF-8: dat is het aantal **bits per byte**. De tekst is voor elk model dezelfde, en zijn bytes ook, wat de tokens ook zijn:
    """)
    return


@app.cell(hide_code=True)
def formula_bpb():
    L.formula(r"\text{bits per byte} = \frac{\sum_i \text{loss}_i}{\ln 2 \cdot \text{bytes}}")
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.md(r"""
    Een model dat niets wist en elk van de 256 waarden van een byte even waarschijnlijk vond, zou 8 bits per byte nodig hebben; een lager getal betekent een beter model van de tekst. Per teken werkt ook voor tekst met gewone letters, zoals in [Perplexity](/course/open/lm-perplexity?lang=nl), maar een teken kan in UTF-8 tot 4 bytes kosten: bytes zijn voor elke tekst dezelfde eenheid.

    ### Nieuwe woorden en tekens

    | Woord of teken | Wat het betekent | In code |
    |---|---|---|
    | **bits per byte** | de totale loss van een tekst in bits, verdeeld over zijn bytes in UTF-8; te vergelijken tussen tokenizers | `sum(losses) / math.log(2) / n_bytes` |
    | $\ln 2$ | de nats in één bit, ongeveer 0,693 | `math.log(2)` |

    ### Recept

    Om base models op een tekst te vergelijken:

    1. Neem een tekst waarop geen van de modellen trainde.
    2. Knip hem in de eigen tokens van elk model, en scoor elk token in vensters van de eigen lengte van het model.
    3. Tel de losses op, maak er bits van, en deel door de bytes van de tekst.
    4. Vergelijk de bits per byte; een perplexity per token alleen tussen dezelfde tokens.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Eén tekst, "The cat sat.", 12 bytes, gescoord door twee modellen met andere tokenizers: de ene knipt hem in 12 tekens, de andere in 4 woorden. De getallen zijn verzonnen.
    """)
    return


@app.cell
def example_bpb():
    _n_bytes = len("The cat sat.".encode("utf-8"))
    _chars = [1.1] * 12    # een loss per teken
    _words = [3.0] * 4     # een loss per woord: elk woord is moeilijker te raden dan één teken
    for _losses in (_chars, _words):
        _mean = sum(_losses) / len(_losses)
        print(len(_losses), round(math.exp(_mean), 1), round(sum(_losses) / math.log(2) / _n_bytes, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Per token lijkt het model van tekens veel beter: een perplexity van 3,0 tegen 20,1. Maar per byte is het model van woorden beter: 1,443 bits tegen 1,587. Het is door de hele tekst iets minder verrast, 12,0 nats tegen 13,2, en alleen de bytes zijn voor allebei gelijk.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_losses():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De loss van elk token

        Maak `token_losses(model, ids, block)` af: stap 2. Loop door `ids` in vensters van `block + 1` ids die één id overlappen: de inputs zijn een venster zonder zijn laatste id, de doelen het venster zonder zijn eerste. Geef de loss van elk doel terug, elk één getal, in een lijst. De check probeert je functie op een klein model waarvan de logits afhangen van hoe ver een token in zijn venster staat, met `ids = [3, 1, 4, 1, 5, 2, 6, 5, 3, 5]` en `block = 4`, en meer.
        """),
        L.task("Laat `token_losses` de loss geven van elk token van `ids` behalve het eerste."),
    ])
    return


@app.cell(hide_code=True)
def exercise_losses():
    box_losses = L.code_box("token-losses", lines=14)
    box_losses
    return (box_losses,)


@app.cell(hide_code=True)
def check_losses(box_losses):
    result_losses = L.check("token-losses", box_losses.value)
    result_losses
    return (result_losses,)


@app.cell(hide_code=True)
def hints_losses():
    L.hints(
        "token-losses",
        [
            "`window = ids[start:start + block + 1]`. Een model neemt een batch, dus `x = torch.tensor([window[:-1]])`, en `model(x)[0]` geeft de logits van dat ene venster: één rij per input.",
            "`F.cross_entropy(logits, y, reduction=\"none\")` geeft een loss per rij in plaats van hun gemiddelde; `.tolist()` maakt er getallen van om aan `losses` toe te voegen.",
        ],
        why="Elk token wordt één keer gescoord, uit hooguit `block` tokens ervoor, zoals het model tekst zag tijdens de training. Het eerste token heeft niets ervoor, dus het krijgt geen loss.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_bpb():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Bits per byte

        Maak `bits_per_byte(losses, pieces)` af: stap 3. `losses` zijn de losses in nats van de gescoorde tokens, `pieces` de tekst waar elk ervan voor staat. Tel de bytes van alle stukken in UTF-8, en geef de totale loss terug in bits per byte. De check roept `bits_per_byte([1.2, 0.7, 2.3], ["The", " café", " is"])` aan, en andere.
        """),
        L.task("Laat `bits_per_byte` losses per token omrekenen naar bits per byte."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bpb():
    box_bpb = L.code_box("bits-per-byte", lines=8)
    box_bpb
    return (box_bpb,)


@app.cell(hide_code=True)
def check_bpb(box_bpb):
    result_bpb = L.check("bits-per-byte", box_bpb.value)
    result_bpb
    return (result_bpb,)


@app.cell(hide_code=True)
def hints_bpb():
    L.hints(
        "bits-per-byte",
        [
            "`piece.encode(\"utf-8\")` geeft de bytes van een stuk, zoals in [Tekst is een rij getallen](/course/open/data-text-as-numbers?lang=nl); `len` telt ze. `é` is één teken, maar twee bytes.",
            "De totale loss is `sum(losses)`, in nats; gedeeld door `math.log(2)` is hij in bits.",
        ],
        why="De bytes van een tekst zijn voor elke tokenizer dezelfde, dus de bits per byte van twee modellen op dezelfde tekst zijn te vergelijken, hoe ze hem ook knippen.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.vstack([
        mo.md(f"""
        ### Je model tegen GPT-2

        De tekst bestaat uit de laatste {C.STORIES} verhalen van `data/tinystories-small.txt`, {C.number(len(C.text().replace(C.B.END, '').encode('utf-8')), 0, L.lang)} bytes, met een end-of-text-token tussen elke twee verhalen: dat staat voor geen tekst, dus het telt 0 bytes, maar zijn loss telt mee, voor beide modellen. De verhalen liggen in de validatieshard uit [Project: een datapipeline voor pretraining](/course/open/llm-data-pipeline?lang=nl), dus je model trainde er nooit op, en GPT-2 (2019) is ouder dan TinyStories (2023). De eerste knop scoort je model uit [Capstone: je eigen kleine LLM pretrainen](/course/open/llm-pretrain-capstone?lang=nl) met je `token_losses`, of het model van de course als je er geen hebt; dat duurt ongeveer een minuut. De tweede downloadt één keer GPT-2 small, 548 MB, zoals in [Kijk in de kansen van een echt LLM op het volgende token](/course/open/lm-peek-real-llm?lang=nl), en scoort het: op de CPU van de course, met één thread, duurde dat ongeveer {round(C.MEASURED['gpt2']['seconds'] / 60)} minuten.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def yours_button_cell():
    yours_button = mo.ui.run_button(label="Scoor je model")
    yours_button
    return (yours_button,)


@app.cell(hide_code=True)
def demo_yours(box_losses, result_losses, yours_button):
    yours = None
    _note = None
    if result_losses.state != "correct":   # geen mo.stop: de tabel na deze cel toont intussen de rijen van de course
        _note = "*Maak eerst oefening 1 af: de demo scoort met je `token_losses`.*"
    elif yours_button.value:
        mo.output.replace(mo.md("**Je model scoren** …"))
        _r = C.score_yours(run_code(box_losses.value).names["token_losses"])
        if _r == "run":
            _note = "*Het scoren stopt met een fout. Komen je modules in `work/projects/llm/` nog door hun checks? Verwijder een bestand dat je veranderde, en check de oefening ervan opnieuw.*"
        elif _r == "losses":
            _note = "*Je `token_losses` stopt met een fout op dit model, of geeft een ander aantal losses dan er tokens te scoren zijn. Werkt hij op een tekst van duizenden tokens?*"
        else:
            yours = _r
    mo.output.replace(mo.md(_note)) if _note else mo.output.clear()   # ook als het scoren klaar is: geen oude voortgangsregel
    return (yours,)


@app.cell(hide_code=True)
def gpt2_button_cell():
    gpt2_button = mo.ui.run_button(label="Scoor GPT-2" if C.gpt2_ready() else "Download en scoor GPT-2 (548 MB)")
    gpt2_button
    return (gpt2_button,)


@app.cell(hide_code=True)
def demo_gpt2(box_losses, gpt2_button, result_losses):
    gpt2 = None
    _note = None
    if result_losses.state != "correct":
        _note = "*Maak eerst oefening 1 af: de demo scoort met je `token_losses`.*"
    elif gpt2_button.value:
        _why = C.download_gpt2(lambda _done, _total: mo.output.replace(mo.md(f"**GPT-2 downloaden** · {round(_done / 1e6)} van {round(_total / 1e6)} MB")))
        if _why is not None:
            _note = {"net": "*Het downloaden lukte niet. Is er verbinding met internet? Druk nog eens op de knop om het opnieuw te proberen.*",
                     "pinned": "*De hub gaf een ander bestand dan de course vastlegde. Probeer het later opnieuw.*",
                     "room": "*Het downloaden stopte: is er op de schijf ruimte voor 548 MB?*"}[_why]
        else:
            mo.output.replace(mo.md("**GPT-2 scoren** · dat duurt een paar minuten …"))
            _r = C.score_gpt2(run_code(box_losses.value).names["token_losses"])
            if _r == "run":
                _note = "*Het scoren van GPT-2 stopt met een fout in je `token_losses`. Werkt hij op een model met vensters van 1.024 tokens?*"
            else:
                gpt2 = _r
    mo.output.replace(mo.md(_note)) if _note else mo.output.clear()
    return (gpt2,)


@app.cell(hide_code=True)
def demo_table(box_bpb, gpt2, result_bpb, yours):
    mo.stop(result_bpb.state != "correct", mo.md("*Maak eerst oefening 2 af: de tabel geeft de bits per byte met je `bits_per_byte`.*"))
    _m = C.MEASURED
    _bpb = run_code(box_bpb.value).names["bits_per_byte"]
    _rows = []
    for _name, _mine, _stand in (("yours", yours, _m["yours"]), ("gpt2", gpt2, _m["gpt2"])):
        if _mine is not None:
            _label = "GPT-2 small" if _name == "gpt2" else "Je model" if _mine["source"] == "yours" else "Het model van de course, op Cosmopedia"
            _value = _bpb(_mine["losses"], _mine["pieces"])
        else:
            _label = "GPT-2 small, de run van de course" if _name == "gpt2" else "De run van de course uit de capstone, op TinyStories"
            _mine, _value = _stand, _stand["bpb"]
        _rows.append(f"| {_label} | **{C.number(_value, 3, L.lang)}** | {C.number(_mine['params'], 0, L.lang)} | {C.number(_mine['tokens'], 0, L.lang)} | {C.number(_mine['loss'], 3, L.lang)} | {C.number(_mine['ppl'], 2, L.lang)} |")
    mo.md("| model | bits per byte | parameters | tokens | loss per token | perplexity |\n|---|---|---|---|---|---|\n" + "\n".join(_rows)
          + "\n\nEen rij van de course staat erin tot je dat model zelf scoort.")
    return


@app.cell(hide_code=True)
def text_exercise_better():
    mo.md(r"""
    ### Oefening 3. Welk model is beter?
    """)
    return


@app.cell(hide_code=True)
def choice_better_cell():
    _m = C.MEASURED
    choice_better = L.choice(
        "which-better",
        question=f"Op deze verhalen heeft de run van de course uit de capstone een perplexity van {C.number(_m['yours']['ppl'], 1, L.lang)} per token en {C.number(_m['yours']['bpb'], 3, L.lang)} bits per byte; GPT-2 small {C.number(_m['gpt2']['ppl'], 1, L.lang)} per token en {C.number(_m['gpt2']['bpb'], 3, L.lang)} bits per byte. Welk model voorspelt deze verhalen beter?",
        options={
            "small": "Het kleine model: zijn perplexity is lager",
            "cannot": "Geen van beide: hun tokenizers verschillen, dus ze zijn niet te vergelijken",
            "right": "GPT-2: het heeft minder bits per byte nodig voor dezelfde tekst",
        },
        correct="right",
        why={
            "small": "Zijn tokens zijn kleiner, dus elke gok ervan is makkelijker: een perplexity per token is alleen te vergelijken tussen dezelfde tokens.",
            "cannot": "Per token niet, maar de tekst en zijn bytes zijn voor allebei gelijk: bits per byte vergelijken ze.",
            "right": f"GPT-2 is door de hele tekst iets minder verrast. Met zo'n 200 keer zoveel parameters, getraind op webpagina's, is het per byte maar {C.number(100 * (1 - _m['gpt2']['bpb'] / _m['yours']['bpb']), 0, L.lang)}% beter op verhalen zoals die waarop het kleine model trainde.",
        },
    )
    choice_better
    return (choice_better,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De perplexity uit [Perplexity](/course/open/lm-perplexity?lang=nl), nu naast bits per byte; het base model uit [Base model of chat model?](/course/open/lm-base-vs-chat?lang=nl), getest op hoe goed het tekst voorspelt, niet op vragen; en je model uit [Capstone: je eigen kleine LLM pretrainen](/course/open/llm-pretrain-capstone?lang=nl), nu naast GPT-2. Latere lessen leren je model instructies te volgen, en meten dat met checks die een programma kan nagaan.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_losses, result_bpb, choice_better):
    L.wrap_up(result_losses, result_bpb, choice_better)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
