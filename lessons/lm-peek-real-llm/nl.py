# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les lm-peek-real-llm. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("lm-peek-real-llm", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [De bigram-tabel als matrix van kansen](/course/open/lm-bigram-matrix?lang=nl) en [Veel klassen: softmax](/course/open/ml-softmax?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_row():
    quick_row = L.choice(
        "quick-row",
        question="Wat staat er in je bigram-tabel van kansen voor Tiny Shakespeare in de rij van \"q\"?",
        options={
            "column": "De kans op \"q\" na elk van de 65 tekens",
            "right": "De kans dat elk van de 65 tekens na \"q\" komt",
            "counts": "Hoe vaak elk teken na \"q\" komt, als hele getallen",
        },
        correct="right",
        why={
            "column": "Dat is de kolom van \"q\": de rij gaat over wat erna komt. Zie [De bigram-tabel als matrix van kansen](/course/open/lm-bigram-matrix?lang=nl).",
            "counts": "Dat zijn de tellingen, de rij van de matrix met tellingen voordat je door zijn som deelt. Zie [De bigram-tabel als matrix van kansen](/course/open/lm-bigram-matrix?lang=nl).",
            "right": "Een kansverdeling over het volgende teken, die samen 1 is. Een echt language model geeft ook zo'n rij, over zijn tokens.",
        },
    )
    quick_row
    return (quick_row,)


@app.cell(hide_code=True)
def quick_check_sum():
    quick_sum = L.choice(
        "quick-sum",
        question="Softmax maakt van de 50.257 logits van een model kansen. Wat is hun som?",
        options={
            "many": "50.257: één voor elk token",
            "logits": "De som van de logits: softmax houdt het totaal gelijk",
            "right": "1: wat de logits ook zijn",
        },
        correct="right",
        why={
            "many": "Elke kans ligt tussen 0 en 1, en softmax deelt door de som van alle e-machten, dus samen zijn ze 1. Zie [Veel klassen: softmax](/course/open/ml-softmax?lang=nl).",
            "logits": "Softmax deelt elke e-macht door de som ervan, dus het totaal is altijd 1, niet de som van de logits. Zie [Veel klassen: softmax](/course/open/ml-softmax?lang=nl).",
            "right": "Softmax deelt elke e-macht door de som ervan. Met 50.257 tokens zijn de meeste kansen piepklein.",
        },
    )
    quick_sum
    return (quick_sum,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je bigram-tabel voorspelt het volgende teken uit één teken: na "e" altijd dezelfde rij, of de tekst nu "the" was of "she". Een echt language model leest de hele tekst ervoor. GPT-2, een open model dat OpenAI in 2019 publiceerde, leerde van 8 miljoen webpagina's, met 124 miljoen parameters. Voor elke tekst berekent het een nieuwe rij: een kans voor elk van de 50.257 tokens van zijn vocabulary. In deze les laad je het en kijk je naar die rij.

    **Een voorbeeld uit het dagelijks leven.** Het toetsenbord van je telefoon stelt drie woorden voor terwijl je typt. Het heeft ook zo'n rij, en laat je de drie woorden met de grootste getallen erin zien.
    """)
    return


@app.cell(hide_code=True)
def predict_capital():
    guess_capital = L.guess(
        "capital",
        question="GPT-2 leest \"The capital of France is\". Welk token krijgt volgens jou de grootste kans om erna te komen?",
        options={
            "paris": "\" Paris\"",
            "right": "\" the\"",
            "france": "\" France\"",
        },
        reveal=f"**\" the\", met {C.share(C.REFERENCE['The capital of France is'][0][1], 'nl')}.** \" Paris\" krijgt {C.share(C.REFERENCE['The capital of France is'][4][1], 'nl')}, op de vijfde plaats, na \" now\", \" a\" en \" France\". GPT-2 beantwoordt geen vraag: het schrijft de tekst verder zoals teksten op het web verdergaan, bijvoorbeeld met \"The capital of France is the city of…\" of \"…is now…\". Welke tokens waarschijnlijk zijn, hangt af van alles wat ervoor staat.",
    )
    guess_capital
    return (guess_capital,)


@app.cell(hide_code=True)
def text_idea():
    _cat = C.REFERENCE["The cat sat on the"]
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een rij voor elke tekst.** GPT-2 maakt van een tekst van $t$ tokens logits met shape $(t, 50257)$: op elke positie een logit voor elk token van zijn vocabulary, voor het token dat na die positie komt. De laatste rij gaat over het token na de hele tekst. Softmax maakt van die rij kansen die samen 1 zijn: de **kansverdeling van het volgende token**. Je bigram-tabel had één vaste rij per teken; GPT-2 berekent zijn rij telkens opnieuw uit alle tokens ervoor, met 12 transformer blocks, die je in latere lessen bouwt. Het leest tot 1.024 tokens terug.

        **De waarschijnlijkste tokens.** Van de 50.257 kansen bekijk je de paar grootste: `torch.topk(probs, k)` geeft de $k$ grootste waarden, `.values`, en hun plaatsen, `.indices`, en dat zijn token IDs. `enc.decode([i])` maakt van token ID `i` weer zijn stuk tekst, zoals in [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl).

        **Eén keer downloaden.** De course levert GPT-2 niet mee: het bestand is 548 MB. De knop in het uitgewerkte voorbeeld downloadt het één keer van Hugging Face, uit `openai-community`, waar Hugging Face de GPT-2 bewaart die OpenAI publiceerde, in een vaste versie, en controleert of elke byte die van het bestand is dat de course vastlegde. Daarna werkt de les zonder verbinding. De licentie is MIT, zoals de model card zegt.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **kansverdeling van het volgende token** | | de kansen die een language model elk token van zijn vocabulary geeft om na een tekst te komen | `torch.softmax(logits[-1], dim=0)` |
        | `torch.softmax(z, dim=0)` | torch softmax | de softmax uit [Veel klassen: softmax](/course/open/ml-softmax?lang=nl) in PyTorch, langs dimensie 0 van `z` | |
        | `torch.topk(p, k)` | torch top k | de $k$ grootste waarden van `p`, in `.values`, en hun plaatsen, in `.indices` | `torch.topk(probs, 10)` |
        """),
        L.watch_out("Het volgende token komt uit de laatste rij, `logits[-1]`. De eerste rij, `logits[0]`, is wat GPT-2 verwacht na alleen het eerste token."),
        mo.md(r"""
        De figuur: de 10 waarschijnlijkste tokens die GPT-2 geeft na "The cat sat on the", elk met zijn kans. Een spatie staat er als ␣: in GPT-2 hoort die bij het token erna. Samen krijgen deze 10 minder dan de helft: de andere 50.247 tokens delen de rest.
        """),
        mo.Html(C.token_bars("The cat sat on the", _cat, {"title": "GPT-2, na \"The cat sat on the\"",
                             "alt": "Tien balken, de waarschijnlijkste eerst: " + ", ".join(f"{C.piece(_t)} {C.share(_p, 'nl')}" for _t, _p in _cat) + "."}, "nl")),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om te zien wat een language model na een tekst verwacht:

    1. De tokens: `ids = enc.encode(text)`.
    2. De logits: `logits = gpt2(torch.tensor(ids))`, binnen `with torch.no_grad():`, met shape $(t, 50257)$.
    3. De kansverdeling van het volgende token: `probs = torch.softmax(logits[-1], dim=0)`.
    4. De waarschijnlijkste tokens: `top = torch.topk(probs, k)`, en `enc.decode([i])` voor elk token ID `i` in `top.indices`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De vier stappen voor de tekst "Once upon a time". `gpt2` is GPT-2 small, dat de cel hieronder laadt zodra het gedownload is.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="GPT-2 downloaden, 548 MB")
    return (get_button,)


@app.cell(hide_code=True)
def gpt2_cell(get_button):
    gpt2 = C.gpt2_or_stop(get_button, {
        "ask": "*GPT-2 staat nog niet op je computer. De knop downloadt het één keer, 548 MB, van Hugging Face. Het uitgewerkte voorbeeld en de cel met je eigen tekst hebben het nodig; de oefeningen niet. Tijdens het downloaden wacht de hele les: doe bij een trage verbinding eerst de oefeningen.*",
        "busy": "**GPT-2 downloaden** · {done} van {total} MB",
        "no_net": "*Het downloaden lukte niet: Hugging Face was niet bereikbaar, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*Het downloaden lukte niet: het bestand dat binnenkwam is niet het bestand dat de course vastlegde, dus de course gooide het weg. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*Het downloaden lukte niet: er is geen ruimte op je schijf voor 548 MB. Maak ruimte vrij, en klik opnieuw.*",
        "failed": "*Het downloaden lukte niet ({why}). Klik opnieuw.*",
        "here": "*GPT-2 small staat op je computer: 124 miljoen parameters, van OpenAI, licentie MIT.*",
    })
    return (gpt2,)


@app.cell
def example_once(gpt2):
    import torch
    import tiktoken

    _enc = tiktoken.get_encoding("gpt2")
    # 1. de tokens van de tekst
    _ids = _enc.encode("Once upon a time")
    print(_ids)
    # 2. de logits: een rij per positie, een logit per token van de vocabulary
    with torch.no_grad():
        _logits = gpt2(torch.tensor(_ids))
    print(_logits.shape)
    # 3. de kansverdeling van het volgende token: de softmax van de laatste rij
    _probs = torch.softmax(_logits[-1], dim=0)
    print(round(_probs.sum().item(), 3))
    # 4. de 5 waarschijnlijkste volgende tokens
    _top = torch.topk(_probs, 5)
    # repr laat het stuk tekst tussen aanhalingstekens zien, zodat je een spatie aan het begin ziet
    for _j in range(5):
        _i = _top.indices[_j].item()
        print(repr(_enc.decode([_i])), round(_top.values[_j].item(), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    _once = C.REFERENCE["Once upon a time"]
    mo.md(f"""
    De tekst heeft 4 tokens, dus de logits hebben 4 rijen van 50.257. De kansen van de laatste rij zijn samen 1. Na "Once upon a time" verwacht GPT-2 het meest een komma, met {C.share(_once[0][1], 'nl')}: in de verhalen die het las, gaat het verder met "Once upon a time, …". Dan komen \" the\", \" I\" en \" he\", elk met minder dan 0,07.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_probs():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De kansverdeling van het volgende token

        Maak `next_probs(logits)` af: stap 3 van het recept, voor logits van elke shape $(t, V)$. De check roept `next_probs(logits)` aan met kleine logits van 2 posities en een vocabulary van 3 tokens, `[[2, 0, 1], [0, 3, 1]]`, zodat het zonder GPT-2 werkt, en vergelijkt de kansen, afgerond op 3 decimalen.
        """),
        L.task("Laat `next_probs` de softmax van de laatste rij van `logits` geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_probs():
    box_probs = L.code_box("next-probs", lines=6)
    box_probs
    return (box_probs,)


@app.cell(hide_code=True)
def check_probs(box_probs):
    result_probs = L.check("next-probs", box_probs.value)
    result_probs
    return (result_probs,)


@app.cell(hide_code=True)
def hints_probs():
    L.hints(
        "next-probs",
        [
            "Het token na de hele tekst komt uit de laatste rij van `logits`.",
            "De laatste rij is `logits[-1]`, een tensor van $V$ logits.",
            "`torch.softmax(___, dim=___)`: de laatste rij, langs zijn enige dimensie.",
        ],
        why="`logits[-1]` is de laatste rij: de logits van het token na de hele tekst. Softmax langs zijn enige dimensie, `dim=0`, maakt er kansen van die samen 1 zijn.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_top():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De waarschijnlijkste tokens

        Maak `top_tokens(probs, k)` af: stap 4 van het recept. De functie geeft de `k` waarschijnlijkste tokens als een lijst van paren: het stuk tekst en zijn kans, het waarschijnlijkste eerst. De check roept `top_tokens(probs, 2)` aan met verzonnen kansen over de 50.257 tokens van GPT-2: 0,5 voor token ID 262, 0,3 voor 257, 0,15 voor 1169 en 0,05 voor 13, en 0 voor alle andere.
        """),
        L.task("Laat `top_tokens` de `k` waarschijnlijkste tokens geven als paren van een stuk tekst en zijn kans."),
    ])
    return


@app.cell(hide_code=True)
def exercise_top():
    box_top = L.code_box("top-tokens", lines=14)
    box_top
    return (box_top,)


@app.cell(hide_code=True)
def check_top(box_top):
    result_top = L.check("top-tokens", box_top.value)
    result_top
    return (result_top,)


@app.cell(hide_code=True)
def hints_top():
    L.hints(
        "top-tokens",
        [
            "`torch.topk(probs, k)` geeft de `k` grootste kansen en hun token IDs.",
            "In de loop is `enc.decode([i])` het stuk tekst van token ID `i`, en `top.values[j].item()` zijn kans.",
            "`top = torch.topk(___, ___)`, en in de loop `pairs.append((enc.decode([___]), top.values[j].item()))`.",
        ],
        why="`torch.topk` geeft de grootste waarden eerst, met hun plaatsen in `probs`: die plaatsen zijn token IDs. `enc.decode` krijgt een lijst van token IDs, dus één ID gaat erin als `[i]`. `.item()` maakt van een tensor met één getal een gewoon getal.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Je eigen tekst

    Typ hieronder een tekst, zodra GPT-2 gedownload is. De cel geeft hem aan GPT-2, met jouw `next_probs` en `top_tokens` zodra ze allebei kloppen, anders die van de course, en laat de 10 waarschijnlijkste volgende tokens zien. GPT-2 leerde van Engelse webpagina's: een Engelse tekst werkt het best.
    """)
    return


@app.cell(hide_code=True)
def prompt_cell():
    prompt_box = mo.ui.text(value="The cat sat on the", label="Je tekst", full_width=True)
    prompt_box
    return (prompt_box,)


@app.cell(hide_code=True)
def demo_text(gpt2, prompt_box, box_probs, result_probs, box_top, result_top):
    _mine = result_probs.state == "correct" and result_top.state == "correct"
    _text = prompt_box.value or "The cat sat on the"
    _d = C.demo(gpt2, _text, box_probs.value if _mine else None, box_top.value if _mine else None)
    _who = "jouw `next_probs` en `top_tokens`" if _d["yours"] else "de `next_probs` en `top_tokens` van de course"
    mo.vstack([
        mo.md(f"Je tekst heeft {C.count(len(_d['pieces']), 'nl')} tokens: " + ("… · " if len(_d["pieces"]) > 40 else "")
              + " · ".join(f"`{C.piece(_p)}`" for _p in _d["pieces"][-40:]) + "."
              + (" GPT-2 leest hooguit 1.024 tokens: het kreeg de laatste 1.024." if _d["cut"] else "") + f" Met {_who}:"),
        mo.Html(C.token_bars(_text, _d["top"], {"title": "GPT-2, na jouw tekst",
                             "alt": "Tien balken, de waarschijnlijkste eerst: " + ", ".join(f"{C.piece(_t)} {C.share(_p, 'nl')}" for _t, _p in _d["top"]) + "."}, "nl")),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_context():
    mo.md(r"""
    ### Oefening 3. Hetzelfde laatste token
    """)
    return


@app.cell(hide_code=True)
def choice_context_cell():
    _cat, _i = dict(C.REFERENCE["The cat sat on the"]), dict(C.REFERENCE["I sat on the"])
    choice_context = L.choice(
        "context",
        question="\"The cat sat on the\" en \"I sat on the\" eindigen allebei op het token \" the\". Een bigram-tabel van tokens geeft na \" the\" voor allebei dezelfde rij. En GPT-2?",
        options={
            "same": "Dezelfde rij: het volgende token hangt af van het laatste token",
            "random": "Elke keer dat je het draait een andere rij, ook voor dezelfde tekst",
            "right": "Voor elk een andere rij: het leest elk token voor \" the\"",
        },
        correct="right",
        why={
            "same": f"GPT-2 leest de hele tekst ervoor. Na \"The cat sat on the\" krijgt \" bed\" {C.share(_cat[' bed'], 'nl')}; na \"I sat on the\" {C.share(_i[' bed'], 'nl')}.",
            "random": "Voor dezelfde tekst geeft GPT-2 elke keer dezelfde logits: er zit geen toeval in. Een token willekeurig kiezen uit de verdeling doet [Sampling: laat het model schrijven](/course/open/lm-sampling?lang=nl).",
            "right": f"Na \"The cat sat on the\": \" floor\" {C.share(_cat[' floor'], 'nl')}, \" bed\" {C.share(_cat[' bed'], 'nl')}. Na \"I sat on the\": \" couch\" {C.share(_i[' couch'], 'nl')}, \" bench\" {C.share(_i[' bench'], 'nl')}, en \" bed\" maar {C.share(_i[' bed'], 'nl')}. In de teksten die GPT-2 las, zitten katten vaker op een bed dan mensen.",
        },
    )
    choice_context
    return (choice_context,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De rij uit [De bigram-tabel als matrix van kansen](/course/open/lm-bigram-matrix?lang=nl), nu berekend voor elke tekst; de logits en softmax uit [Project: hetzelfde bigram-model, geleerd met gradient descent](/course/open/lm-bigram-neural?lang=nl), uit 124 miljoen parameters in plaats van één tabel; de tokens uit [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl); en een model van een hub, zoals in [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl). De transformer blocks in GPT-2 komen in latere lessen, waarin je zelf een GPT bouwt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_probs, result_top, choice_context):
    L.wrap_up(result_probs, result_top, choice_context)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
