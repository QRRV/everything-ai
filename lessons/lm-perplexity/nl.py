# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les lm-perplexity. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("lm-perplexity", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Hoe goed is een language model?](/course/open/lm-nll?lang=nl) en [Cross-entropy en KL-divergentie](/course/open/info-cross-entropy-kl?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_lower():
    quick_lower = L.choice(
        "quick-lower",
        question="Op dezelfde tekst heeft model A een gemiddelde NLL van 2,2 en model B een van 2,5. Welk model voorspelt de tekst beter?",
        options={
            "b": "B: een hoger getal is een betere score",
            "right": "A: het is minder verrast door de tekens die echt kwamen",
            "same": "Geen van beide: 2,2 en 2,5 liggen te dicht bij elkaar om de modellen uit elkaar te houden",
        },
        correct="right",
        why={
            "b": "De NLL telt verrassingen op, min de log van elke kans: hoe hoger de kansen, hoe lager de NLL. Zie [Hoe goed is een language model?](/course/open/lm-nll?lang=nl).",
            "same": "Op dezelfde tekst is een lagere gemiddelde NLL gewoon beter: 0,3 per teken over een hele tekst is een groot verschil. Zie [Hoe goed is een language model?](/course/open/lm-nll?lang=nl).",
            "right": "Lager is beter: A gaf de tekens die kwamen hogere kansen. Vandaag maak je van zo'n getal een getal dat je je kunt voorstellen.",
        },
    )
    quick_lower
    return (quick_lower,)


@app.cell(hide_code=True)
def quick_check_floor():
    quick_floor = L.choice(
        "quick-floor",
        question="Tekst komt uit een bron met entropie $H(p)$. Een model $q$ wordt erop getraind. Hoe laag kan zijn cross-entropy $H(p, q)$ worden?",
        options={
            "zero": "Tot 0: met genoeg training is het model nooit meer verrast",
            "kl": "Tot de KL-divergentie tussen $p$ en $q$: training haalt de entropie weg",
            "right": "Tot $H(p)$: de KL-divergentie erbovenop is nooit negatief",
        },
        correct="right",
        why={
            "zero": "Alleen een bron zonder enige onzekerheid heeft entropie 0; tekst is niet zo. Zie [Cross-entropy en KL-divergentie](/course/open/info-cross-entropy-kl?lang=nl).",
            "kl": "De KL-divergentie is wat het model aan verrassing bovenop de entropie toevoegt, en die is nooit negatief. Zie [Cross-entropy en KL-divergentie](/course/open/info-cross-entropy-kl?lang=nl).",
            "right": "$H(p, q) = H(p) + D_{\\mathrm{KL}}(p \\,\\|\\, q)$, en de KL-divergentie is minstens 0: de entropie van de tekst is een bodem waar geen model onder komt. Perplexity heeft die bodem ook.",
        },
    )
    quick_floor
    return (quick_floor,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een gemiddelde NLL van 2,20 kun je je moeilijk voorstellen. Is dat goed? Hoeveel beter is 2,20 dan 2,45? Papers en model cards geven language models een ander getal, gemaakt van dezelfde loss: **perplexity**. Dat beantwoordt een vraag die je je wel kunt voorstellen: tussen hoeveel tokens kiest het model, alsof het er gelijkmatig tussen koos? Een model met een perplexity van 9 wordt gemiddeld even sterk verrast door elk token als door de worp van een dobbelsteen met 9 kanten.
    """)
    return


@app.cell(hide_code=True)
def predict_ratio():
    guess_ratio = L.guess(
        "lower-loss",
        question="Model A heeft een gemiddelde NLL van 2,2 per teken, model B een van 2,4: de loss van A is ongeveer 8% lager. Hoeveel lager is de perplexity van A?",
        options={
            "same": "Ook ongeveer 8% lager",
            "right": "Ongeveer 18% lager",
            "less": "Ongeveer 3% lager",
        },
        reveal="**Ongeveer 18% lager.** Perplexity is e tot de macht van de loss: $e^{2{,}2} \\approx 9{,}0$ en $e^{2{,}4} \\approx 11{,}0$. Een loss die 0,2 lager is, deelt de perplexity door $e^{0{,}2} \\approx 1{,}22$, wat de loss ook was: op de schaal van perplexity telt elke stap omlaag in loss als een vaste factor.",
    )
    guess_ratio
    return (guess_ratio,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Geef het model een tekst van $n$ tokens en neem de gemiddelde NLL, zoals in [Hoe goed is een language model?](/course/open/lm-nll?lang=nl): het gemiddelde van min de log van de kans die het gaf aan elk token dat echt kwam. **Perplexity** is e tot de macht daarvan:
        """),
        L.formula(r"\begin{aligned} [[loss:\mathrm{PPL}]] &= e^{[[loss:\text{gemiddelde NLL}]]} \\ &= \exp\Big(-\frac{1}{n} \sum_{t=1}^{n} \log [[pred:P(x_t \mid x_{<t})]]\Big) \end{aligned}"),
        mo.md(r"""
        Anders gezegd: geef elk token één en dezelfde kans, zo gekozen dat de gemiddelde NLL gelijk blijft; de perplexity is 1 gedeeld door die kans, en die kans heet het meetkundig gemiddelde van de kansen. Een model dat elk van 65 tekens de kans $1/65$ geeft, heeft een perplexity van precies 65; een model dat altijd zeker is en altijd gelijk heeft, 1. Met $\log_2$ en 2 tot de macht krijg je hetzelfde getal: het grondtal valt weg.

        Perplexity is per token, dus hij hangt af van de tokenizer. Een model dat in tekens telt en een model dat telt in de tokens uit [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl) kiezen tussen andere dingen, en hun perplexities kun je niet zomaar vergelijken. Om te vergelijken verdeel je de totale NLL van de tekst over dezelfde eenheid, zoals zijn tekens: de gemiddelde NLL per token, keer het aantal tokens, gedeeld door het aantal tekens. En perplexity heeft een bodem: de cross-entropy van een model is minstens de entropie van de tekst, dus zijn perplexity is minstens $e^{H(p)}$, met $H$ in nats ($2^{H(p)}$ met $H$ in bits). Die bodem geldt op nieuwe tekst; op de tekst waarop het getraind is, kan een model eronder komen.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **perplexity** | | e tot de macht van de gemiddelde NLL: het aantal tokens waartussen een model kiest, alsof het gelijkmatig koos | `np.exp(nll)` |
        | $\mathrm{PPL}$ | P P L | perplexity | `ppl` |
        | $x_{<t}$ | x vóór t | de tokens vóór token $t$: $x_1$ tot en met $x_{t-1}$ | `tokens[:t]` |
        """),
        L.watch_out("Een perplexity zegt alleen iets samen met zijn tekst en zijn tokens. Op de tekst waarop het getraind is, lijkt een model beter dan op nieuwe tekst; en een perplexity per GPT-2-token is geen perplexity per teken. Vergelijk modellen op dezelfde onbekende tekst, in dezelfde eenheid."),
        mo.md(r"""
        De figuur: de kromme van gemiddelde NLL naar perplexity, met drie modellen erop. Het getelde bigram-model van Tiny Shakespeare heeft op de tekst die het telde een gemiddelde NLL van 2,45: een perplexity van 11,6, een keuze uit ongeveer 12 tekens. Een model dat alle 65 tekens dezelfde kans geeft, zit op 65, en een model dat altijd zeker is en gelijk heeft op 1. De kromme stijgt steeds sneller: aan de hoge kant betekent iets meer loss veel meer keuzes.
        """),
        mo.Html(C.ppl_picture({"title": "e tot de macht van de loss", "x": "gemiddelde NLL", "y": "perplexity", "perfect": "altijd goed: {}",
                               "bigram": "bigram-model: {}", "uniform": "alle 65 gelijk: {}", "decimal": ",",
                               "alt": "De kromme van perplexity, e tot de macht van de gemiddelde NLL, van 1 bij een loss van 0 tot 65 bij een loss van 4,17. Erop: altijd goed, 1; het bigram-model, 11,6 bij een loss van 2,45; en alle 65 tekens even waarschijnlijk, 65."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    De perplexity van een model op een tekst:

    1. Voor elk token van de tekst de kans die het model het gaf, gegeven de tokens ervoor.
    2. De gemiddelde NLL: min het gemiddelde van de logs van die kansen.
    3. De perplexity: e tot de macht van de gemiddelde NLL.
    4. Vergelijk alleen op dezelfde onbekende tekst, in dezelfde eenheid; verschillen de tokens, verdeel dan eerst de totale NLL over de tekens.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het getelde bigram-model op "the king": de kansen van zijn 7 paren, de gemiddelde NLL en de perplexity. `C.bigram_prob(a, b)` geeft de kans op `b` na `a`. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 3 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_perplexity():
    import numpy as np

    _text = "the king"
    # 1. de kans op elk teken na het teken ervoor
    _probs = [C.bigram_prob(_text[_i - 1], _text[_i]) for _i in range(1, len(_text))]
    print(np.round(_probs, 4))
    # 2. de gemiddelde NLL
    _nll = -np.mean(np.log(_probs))
    print(round(_nll, 4))
    # 3. de perplexity
    _ppl = _nll  # maak af: e tot de macht _nll
    print(round(_ppl, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De kansen lopen van 0,355 voor "e" na "h" tot 0,0094 voor "k" na een spatie: weinig woorden beginnen met "k". Hun gemiddelde NLL is 1,9525. Zoals de cel nu is, print de laatste regel dat nog eens, `1.952`: de loss, niet de perplexity. Afgemaakt, met `np.exp(_nll)`, print hij `7.046`: op "the king" is het bigram-model zo onzeker als een keuze uit ongeveer 7 tekens, minder dan zijn 11,6 op de hele tekst, omdat zijn paren "th", "he" en "e " waarschijnlijk zijn.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_compare():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Twee perplexities
    """)
    return


@app.cell(hide_code=True)
def choice_compare_cell():
    choice_compare = L.choice(
        "compare",
        question="Op dezelfde onbekende tekst heeft model A een perplexity van 9 per teken, en model B een van 40 per GPT-2-token. Welk model is beter?",
        options={
            "a": "A: 9 is veel lager dan 40",
            "b": "B: het kiest bij elke stap tussen meer tokens, dus het weet meer van de tekst",
            "right": "Dat volgt niet uit deze getallen: breng ze eerst naar dezelfde eenheid",
        },
        correct="right",
        why={
            "a": "Lager is alleen beter in dezelfde eenheid. Een GPT-2-token is vaak een paar tekens, dus B maakt voor dezelfde tekst minder keuzes, elk uit meer mogelijkheden.",
            "b": "Een hogere perplexity betekent meer onzekerheid per stap, niet meer kennis; maar B zet ook minder stappen.",
            "right": "Verdeel de totale NLL van B over de tekens van de tekst: bij 4 tekens per token is een perplexity van 40 per token $40^{1/4} \\approx 2{,}5$ per teken, en dan is B de betere.",
        },
    )
    choice_compare
    return (choice_compare,)


@app.cell(hide_code=True)
def text_exercise_perplexity():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Perplexity

        Maak `perplexity(probs)` af: die krijgt de kansen die een model gaf aan de tokens die echt kwamen, als lijst, en geeft de perplexity. Vervang de twee `...`, volgens stap 2 en 3 van het recept. De check roept `perplexity([0.5, 0.125])` aan, wat `4.0` moet geven, en probeert andere lijsten, zoals een met elke kans $1/65$.
        """),
        L.task("Laat `perplexity` e tot de macht van de gemiddelde NLL van een lijst kansen geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_perplexity():
    box_perplexity = L.code_box("perplexity", lines=7)
    box_perplexity
    return (box_perplexity,)


@app.cell(hide_code=True)
def check_perplexity(box_perplexity):
    result_perplexity = L.check("perplexity", box_perplexity.value)
    result_perplexity
    return (result_perplexity,)


@app.cell(hide_code=True)
def hints_perplexity():
    L.hints(
        "perplexity",
        [
            "Twee regels uit het uitgewerkte voorbeeld, met `probs` in plaats van `_probs`: de gemiddelde NLL, en de afgemaakte stap 3.",
            "De NLL heeft `np.log` van de kansen nodig, hun `np.mean`, en een min; de perplexity is `np.exp` van de NLL.",
        ],
        why="Voor `[0.5, 0.125]` zijn de logs −0,693 en −2,079, dus de gemiddelde NLL is 1,386, en $e^{1{,}386} \\approx 4$: één gedeeld door het meetkundig gemiddelde van 0,5 en 0,125, dat $\\sqrt{0{,}0625} = 0{,}25$ is.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_char():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Per teken

        Maak `char_perplexity(avg_nll, n_tokens, n_chars)` af: een model haalde op een tekst van `n_chars` tekens, geknipt in `n_tokens` tokens, een gemiddelde NLL van `avg_nll` per token. De functie geeft zijn perplexity per teken. Vervang de `...` door de NLL per teken, volgens stap 4 van het recept. De check roept `char_perplexity(3.0, 100, 400)` aan, wat ongeveer `2.117` moet geven, en probeert andere teksten.
        """),
        L.task("Laat `char_perplexity` van een gemiddelde NLL per token een perplexity per teken maken."),
    ])
    return


@app.cell(hide_code=True)
def exercise_char():
    box_char = L.code_box("char-perplexity", lines=6)
    box_char
    return (box_char,)


@app.cell(hide_code=True)
def check_char(box_char):
    result_char = L.check("char-perplexity", box_char.value)
    result_char
    return (result_char,)


@app.cell(hide_code=True)
def hints_char():
    L.hints(
        "char-perplexity",
        [
            "De totale NLL van de tekst is dezelfde, hoe je ook telt: het gemiddelde per token keer het aantal tokens.",
            "Deel dat totaal door het aantal tekens, en `np.exp` in de laatste regel doet de rest.",
        ],
        why="Voor `(3.0, 100, 400)` is de totale NLL 3,0 × 100 = 300, dus 0,75 per teken, en $e^{0{,}75} \\approx 2{,}117$. De perplexity per token, $e^{3} \\approx 20{,}1$, is dat getal tot de macht 4: vier tekens per token.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De gemiddelde NLL uit [Hoe goed is een language model?](/course/open/lm-nll?lang=nl) is de cross-entropy uit [Cross-entropy en KL-divergentie](/course/open/info-cross-entropy-kl?lang=nl) tussen de tekst en het model; perplexity is e tot de macht daarvan, een aantal keuzes. Zijn bodem is $e^{H(p)}$, bepaald door de entropie van de tekst zelf. Elk language model vanaf hier, ook je GPT, krijgt een perplexity op onbekende tekst, en voor modellen met andere tokens breng je die eerst naar tekens, zoals in oefening 3. De volgende les, *Convolutie in detail: kernel size, stride en padding*, gaat terug naar afbeeldingen: wat een convolutional layer doet met zijn kernel, hoe ver hij stapt, en wat er aan de randen gebeurt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_compare, result_perplexity, result_char):
    L.wrap_up(choice_compare, result_perplexity, result_char)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
