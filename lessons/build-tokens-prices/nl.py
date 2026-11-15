# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-tokens-prices. Route: course/route/lessen/bouwen.yaml. Checks, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-tokens-prices", "nl")
    C = L.checks   # de tellingen en antwoorden van Qwen3 van de course, en de demo die telt voordat hij verstuurt


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl), en één over [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_count():
    quick_count = L.choice(
        "quick-count",
        question="Wat geeft `len(enc.encode(text))`, met een tokenizer `enc`?",
        options={
            "chars": "Het aantal tekens van `text`",
            "words": "Het aantal woorden van `text`",
            "tokens": "Het aantal tokens van `text`",
        },
        correct="tokens",
        why={
            "tokens": "`enc.encode(text)` geeft de lijst van token IDs, één per stuk dat de tokenizer knipte, en `len` telt ze.",
            "chars": "Dat is `len(text)`. Wat geeft `encode` terug? Zie [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl).",
            "words": "Dat is `len(text.split())`; een tokenizer knipt sommige woorden in meer stukken. Zie [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl).",
        },
    )
    quick_count
    return (quick_count,)


@app.cell(hide_code=True)
def quick_check_role():
    quick_role = L.choice(
        "quick-role",
        question="Wat staat er in een bericht met de rol `\"system\"`?",
        options={
            "answer": "Het antwoord van het model",
            "instruction": "Instructies voor het hele gesprek, zoals hoe het model moet antwoorden",
            "question": "De vraag van de gebruiker",
        },
        correct="instruction",
        why={
            "instruction": "De system prompt komt eerst en geldt voor elke beurt: wie het model is, en hoe het antwoordt.",
            "answer": "Het antwoord van het model heeft de rol `\"assistant\"`. Zie [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl).",
            "question": "De vraag heeft de rol `\"user\"`. Zie [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl).",
        },
    )
    quick_role
    return (quick_role,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een gehoste API stuurt zijn rekening per token, niet per vraag. Een antwoord van 500 tokens kost meer dan een van 50, en de tokens die eruit komen kosten meestal een paar keer zoveel als die erin gaan. Als je telt voordat je verstuurt, weet je wat een onderdeel van je app kost, en waar je kunt besparen: een kortere prompt, een korter antwoord, of een goedkoper model.

    **Een voorbeeld uit het dagelijks leven.** Bij een copyshop betaal je per pagina, en een pagina in kleur kost meer dan een in zwart-wit. Tel de pagina's voordat je print, en je weet de rekening. Dat is maar een vergelijking; hieronder tel je echte tokens.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **input token** | een token van wat je stuurt: de system prompt, het gesprek tot nu toe, de vraag; het antwoord telt ze in `usage` als `prompt_tokens` |
    | **output token** | een token dat het model schrijft; geteld als `completion_tokens`, en meestal duurder, omdat het model ze een voor een schrijft |
    | **tokenprijs** | wat een API vraagt per miljoen tokens, één prijs voor input tokens en één voor output tokens; in deze les verzonnen prijzen van 3 en 15 dollar |
    | **chat template** | de paar eigen tokens die de server om elk bericht zet, zodat het model ziet wie wat zegt; ook die tellen als input tokens |

    ### Elk model telt in zijn eigen tokens

    De course telde vijf teksten met de tokenizer van GPT-2, uit [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl), en met die van Qwen3:
    """)
    return


@app.cell(hide_code=True)
def picture_counts():
    mo.Html(C.counts_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_counts():
    _n = C.numbers(L.lang)
    mo.md(rf"""
    De Engelse zin is voor allebei {_n['en_qwen']} tokens. Dezelfde zin in het Nederlands is {_n['nl_gpt2']} tokens voor GPT-2 en {_n['nl_qwen']} voor Qwen3, met minder woorden: beide tokenizers leerden meer Engels dan Nederlands, en die van GPT-2 bijna niets anders. Bij het getal is het andersom: {_n['number_qwen']} tokens voor Qwen3 tegen {_n['number_gpt2']} voor GPT-2, omdat Qwen3 cijfers een voor een knipt. Je telt dus met de tokenizer van het model dat je aanroept; een andere is alleen een schatting.
    """)
    return


@app.cell(hide_code=True)
def text_answers():
    mo.md(r"""
    ### Wat eruit komt, kost het meest

    De course stelde Qwen3 via zijn eigen server twee vragen, een die om een kort antwoord vraagt en een die om een lang antwoord vraagt, en las de tokens in `usage`:
    """)
    return


@app.cell(hide_code=True)
def picture_answers():
    mo.Html(C.answers_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_bill():
    _n = C.numbers(L.lang)
    mo.vstack([
        mo.md(rf"""
        Het lange antwoord heeft {_n['long_out']} tokens eruit tegen {_n['short_out']}, {_n['times']} keer zoveel, en het kostte meer dan tien keer zoveel. Het duurde ook {_n['long_seconds']} seconden in plaats van {_n['short_seconds']}: het model schrijft de tokens die eruit komen een voor een. De vraag zelf is {_n['short_text']} tokens voor de tokenizer van Qwen3, maar de server telde {_n['short_in']} tokens erin: het chat template voegde er {_n['template']} toe.

        Een aanroep kost `tokens_in * price_in / 1_000_000 + tokens_out * price_out / 1_000_000`; `1_000_000` is een miljoen, en Python slaat de liggende streepjes over. Voordat je verstuurt, weet je de tokens erin, door te tellen, maar niet de tokens eruit. Je weet wel het meeste: het antwoord stopt bij `max_tokens`.
        """),
        L.watch_out("Elk bericht van het gesprek gaat met elke aanroep mee, en telt opnieuw als input tokens. Een lang gesprek wordt met elke beurt duurder."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: wat een aanroep kost, voordat je hem verstuurt

    1. **Tel de prompt:** `len(enc.encode(prompt))`, met de tokenizer van het model dat je aanroept; de server voegt per bericht een paar tokens toe.
    2. **Begrens het antwoord:** met `max_tokens` heeft het antwoord hooguit zoveel tokens.
    3. **Reken beide:** tokens erin keer de prijs erin, plus tokens eruit keer de prijs eruit, gedeeld door een miljoen.
    4. **Lees daarna de echte telling:** `usage` in het antwoord heeft `prompt_tokens` en `completion_tokens`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het meeste dat een aanroep met de vraag *Name three colors of the rainbow.* kan kosten, met hooguit 50 tokens eruit, voor een model met de tokenizer van GPT-2:

    ```python
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    prompt = "Name three colors of the rainbow."
    tokens_in = len(enc.encode(prompt))
    tokens_in * 3 / 1_000_000 + 50 * 15 / 1_000_000
    ```

    1. **De tokenizer:** `enc` knipt tekst zoals GPT-2 dat doet, uit de eigen kopie van de course.
    2. **De prompt:** `enc.encode(prompt)` geeft 7 token IDs, dus `tokens_in` is 7.
    3. **De tokens erin:** 7 keer 3 dollar per miljoen is 21 miljoenste dollar.
    4. **Het meeste voor het antwoord:** 50 tokens tegen 15 dollar per miljoen is 750 miljoenste; samen `0.000771` dollar. Het echte antwoord kan korter zijn, en minder kosten.
    """)
    return


@app.cell(hide_code=True)
def text_cost():
    _f = C.findings()
    mo.vstack([
        mo.md(rf"""
        ## Zelf doen

        ### Oefening 1. Wat een aanroep kost

        Schrijf `cost(tokens_in, tokens_out, price_in, price_out)`: wat één aanroep kost, in dollar, met de prijzen per miljoen tokens. Het korte antwoord van de course hierboven had {_f['short_in']} tokens erin en {_f['short_out']} eruit; tegen 3 en 15 dollar per miljoen is dat `{_f['short_cost']:.6f}` dollar:

        ```python
        cost({_f['short_in']}, {_f['short_out']}, 3, 15)
        ```
        """),
        L.task("Maak `cost` af: de tokens erin tegen `price_in` plus de tokens eruit tegen `price_out`, per miljoen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cost():
    box_cost = L.code_box("cost", lines=3)
    box_cost
    return (box_cost,)


@app.cell(hide_code=True)
def check_cost(box_cost):
    result_cost = L.check("cost", box_cost.value)
    result_cost
    return (result_cost,)


@app.cell(hide_code=True)
def hints_cost():
    L.hints(
        "cost",
        [
            "Twee delen, elk met een eigen prijs: wat erin gaat, en wat eruit komt.",
            "Het deel dat erin gaat is `tokens_in * price_in / 1_000_000`; het deel dat eruit komt is hetzelfde met `tokens_out` en `price_out`.",
            "`return tokens_in * price_in / 1_000_000 + ...`, met het deel voor de tokens eruit op de plek van de puntjes.",
        ],
        why="""```python
def cost(tokens_in, tokens_out, price_in, price_out):
    return tokens_in * price_in / 1_000_000 + tokens_out * price_out / 1_000_000
```

Elke soort token tegen zijn eigen prijs, en de prijzen zijn per miljoen tokens.""",
    )
    return


@app.cell(hide_code=True)
def text_most():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Het meeste, voordat je verstuurt

        Schrijf `most(prompt, max_tokens, price_in, price_out)`: het meeste dat een aanroep met `prompt` kan kosten, als het antwoord tot `max_tokens` tokens mag hebben. Tel de prompt met `enc`, de tokenizer van GPT-2; het chat template mag je weglaten. Voor het uitgewerkte voorbeeld geeft `most("Name three colors of the rainbow.", 50, 3, 15)` `0.000771`.
        """),
        L.task("Maak `most` af: tel de tokens van de prompt met `enc`, neem `max_tokens` voor het antwoord, en reken beide."),
    ])
    return


@app.cell(hide_code=True)
def exercise_most():
    box_most = L.code_box("most", lines=9)
    box_most
    return (box_most,)


@app.cell(hide_code=True)
def check_most(box_most):
    result_most = L.check("most", box_most.value)
    result_most
    return (result_most,)


@app.cell(hide_code=True)
def hints_most():
    L.hints(
        "most",
        [
            "Stap 1 tot 3 van het recept: tellen, begrenzen, rekenen.",
            "`tokens_in = len(enc.encode(prompt))`, en het antwoord heeft hooguit `max_tokens` tokens.",
            "`tokens_out = max_tokens`, en dan `return tokens_in * price_in / 1_000_000 + tokens_out * price_out / 1_000_000`.",
        ],
        why="""```python
def most(prompt, max_tokens, price_in, price_out):
    tokens_in = len(enc.encode(prompt))
    tokens_out = max_tokens
    return tokens_in * price_in / 1_000_000 + tokens_out * price_out / 1_000_000
```

De prompt kun je tellen voordat je hem verstuurt; het antwoord kun je alleen begrenzen, dus reken je het meeste ervan.""",
    )
    return


@app.cell(hide_code=True)
def text_cheaper():
    mo.md(r"""
    ### Oefening 3. Waar je bespaart
    """)
    return


@app.cell(hide_code=True)
def choice_cheaper_cell():
    choice_cheaper = L.choice(
        "cheaper",
        question="Een onderdeel van je app stuurt 1.000 tokens erin en krijgt er 500 terug, tegen 3 en 15 dollar per miljoen. Welke verandering bespaart het meest?",
        options={
            "same": "Allebei evenveel: 200 tokens zijn 200 tokens",
            "prompt": "Een prompt die 200 tokens korter is",
            "answer": "Een antwoord dat 200 tokens korter is",
        },
        correct="answer",
        why={
            "answer": "200 tokens eruit tegen 15 dollar per miljoen besparen 0,003 dollar; 200 tokens erin tegen 3 maar 0,0006 dollar. Vragen om een kort antwoord, of een lagere `max_tokens`, bespaart het meest.",
            "same": "Een token erin en een token eruit hebben een andere prijs. Welke is duurder?",
            "prompt": "Dat bespaart 200 tokens tegen 3 dollar per miljoen. Wat besparen 200 tokens tegen de andere prijs?",
        },
    )
    choice_cheaper
    return (choice_cheaper,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Tel met de eigen tokenizer van Qwen3, en verstuur

    Schrijf een prompt: de course telt hem met de tokenizer van Qwen3 en met die van GPT-2, stuurt hem via zijn eigen server naar Qwen3-0.6B met hooguit 200 tokens eruit, en laat zien wat de server telde, en wat dat kost tegen de verzonnen prijzen.
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
def prompt_box_cell():
    prompt_box = mo.ui.text_area(value="Leg in twee zinnen uit waarom de lucht blauw is.", label="Je prompt", full_width=True)
    send_button = mo.ui.run_button(label="Tel hem, en verstuur hem")
    mo.vstack([prompt_box, send_button])
    return prompt_box, send_button


@app.cell(hide_code=True)
def demo_answer(prompt_box, send_button):
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif not send_button.value:
        _out = mo.md("*Druk op de knop om je prompt te tellen en te versturen.*")
    elif not prompt_box.value.strip():
        _out = mo.md("*Schrijf eerst een prompt in het vak.*")
    else:
        mo.output.replace(mo.md("*Geteld; de server laadt Qwen3, dat het antwoord schrijft …*"))
        _words = {"title": "Je prompt, geteld en verstuurd",
                  "done": "Vóór het versturen: {counted} tokens voor de tokenizer van Qwen3 ({gpt2} voor die van GPT-2), en {template} voor de chat template, dus hooguit ${worst} met {most} tokens eruit. De server telde {sent} tokens erin en {out} eruit: ${real}, na {seconds} seconden.",
                  "failed": "De server antwoordde niet ({why}). Druk opnieuw op de knop.",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.demo(prompt_box.value, 200, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Je telt met de tokenizers uit [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl), en elk bericht uit [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl), ook de system prompt, telt als input tokens. In de volgende les, [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl), verstuur je het request zelf, en lees je `usage` in het antwoord.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_cheaper, result_cost, result_most):
    L.wrap_up(result_cost, result_most, choice_cheaper)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
