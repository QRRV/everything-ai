# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les lm-top-k-top-p. Route: course/route/lessen/taal.yaml. Checks: checks.py.
# termen:allow nucleus sampling

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("lm-top-k-top-p", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Temperature: gedurfder of veiliger tekst](/course/open/lm-temperature?lang=nl) en [Tuples, sets en sorteren](/course/open/py-tuples-sets?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_hot():
    quick_hot = L.choice(
        "quick-hot",
        question="Bij een temperature van 2 wordt de tekst van je model vreemder. Waarom?",
        options={
            "right": "De onwaarschijnlijke tekens krijgen een groter deel",
            "never": "Tekens die nooit kwamen, krijgen ook een kans",
            "random": "Het model gebruikt zijn tabel niet meer en trekt willekeurig",
        },
        correct="right",
        why={
            "never": "Een teken dat nooit kwam, houdt bij elke temperature kans 0. Zie [Temperature: gedurfder of veiliger tekst](/course/open/lm-temperature?lang=nl).",
            "random": "Het model trekt nog steeds uit zijn tabel, alleen met vlakkere kansen. Zie [Temperature: gedurfder of veiliger tekst](/course/open/lm-temperature?lang=nl).",
            "right": "Boven 1 worden de verschillen tussen de logits kleiner, dus worden de onwaarschijnlijke tekens vaker getrokken.",
        },
    )
    quick_hot
    return (quick_hot,)


@app.cell(hide_code=True)
def quick_check_sorted():
    quick_sorted = L.choice(
        "quick-sorted",
        question="Wat geeft `sorted([0.2, 0.5, 0.3])[-1]`?",
        options={
            "first": "`0.2`: het eerste getal van de lijst",
            "last": "`0.3`: het laatste getal van de lijst",
            "right": "`0.5`: het grootste getal, na het sorteren achteraan",
        },
        correct="right",
        why={
            "first": "`sorted` zet de getallen eerst op volgorde, van klein naar groot, en `[-1]` is de laatste daarvan. Zie [Tuples, sets en sorteren](/course/open/py-tuples-sets?lang=nl).",
            "last": "`[-1]` is het laatste getal van de gesorteerde lijst, niet van de lijst die je meegaf. Zie [Tuples, sets en sorteren](/course/open/py-tuples-sets?lang=nl).",
            "right": "`sorted` geeft `[0.2, 0.3, 0.5]`, van klein naar groot, en `[-1]` is de laatste daarvan: de grootste.",
        },
    )
    quick_sorted
    return (quick_sorted,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Bij sampling kan elk teken worden getrokken dat ooit na het laatste teken kwam, ook een dat daar maar één keer in alle toneelstukken stond. Elk van die zeldzame trekkingen heeft een kleine kans, maar je model trekt opnieuw voor elk teken, dus ze tellen op: een hoofdletter midden in een woord, een vraagteken na één letter. Een temperature onder 1 maakt ze zeldzamer, maar nooit onmogelijk. Een andere manier is de onwaarschijnlijke tekens helemaal af te snijden, en alleen uit de waarschijnlijke te trekken. Language models doen dat vaak als ze schrijven.

    **Een voorbeeld uit het dagelijks leven.** Een klein restaurant met een lange kaart kookt op een drukke avond alleen de gerechten die het vaakst besteld worden. Af en toe had een gast liever iets anders gehad, maar de keuken blijft snel en er komt niets vreemds uit.
    """)
    return


@app.cell(hide_code=True)
def predict_space():
    guess_space = L.guess(
        "keep-space",
        question=f"Na een spatie kwamen in de toneelstukken {C.possible_after(' ')} verschillende tekens voor. Je houdt alleen de waarschijnlijkste, zo weinig mogelijk, die samen 90% van de kans bevatten. Hoeveel denk je dat je er houdt?",
        options={
            "most": "Ongeveer 50: de zeldzame tellen nauwelijks op",
            "right": "Ongeveer 20: zo'n derde van alle tekens",
            "few": "Ongeveer 5: de paar bovenste hebben al bijna alles",
        },
        reveal=f"**Ongeveer 20: {C.kept_after(' ', 0.9)} van de {C.possible_after(' ')}.** Na een spatie is de kans over veel letters verspreid, dus zijn er {C.kept_after(' ', 0.9)} nodig om 90% te halen. De andere {C.possible_after(' ') - C.kept_after(' ', 0.9)} bevatten samen {L.number(100 * C.tail_share(' '), 1)}% van de kans: bij gewone sampling is na een spatie ongeveer één op de {round(1 / C.tail_share(' '))} trekkingen een van die tekens. Na \"q\" heeft één teken al alles.",
    )
    guess_space
    return (guess_space,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Top-k sampling** houdt alleen de k waarschijnlijkste tekens van de rij. De andere krijgen kans 0, en de k die overblijven worden gedeeld door hun som, zodat ze weer optellen tot 1. Daarna trek je zoals eerder. Met k = 5 trekt het model altijd een van de vijf waarschijnlijkste tekens.

        **Top-p sampling** houdt de waarschijnlijkste tekens, zo weinig mogelijk, die samen minstens kans p bevatten, bijvoorbeeld 0,9. Het heet ook wel nucleus sampling. Hoeveel het er houdt, hangt af van de rij: na "q" heeft één teken alles, dus houdt top-p er één; na een spatie, waar de kans verspreid is, houdt het er veel. Top-k houdt er k, na een spatie net zo goed als na "e", hoe verspreid de rij ook is.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **top-k sampling** | top k | alleen trekken uit de k waarschijnlijkste tokens: de rest krijgt kans 0, en de tokens die overblijven worden gedeeld door hun som | `cut = sorted(p)[-k]` |
        | **top-p sampling** | top p | alleen trekken uit de waarschijnlijkste tokens, zo weinig mogelijk, die samen minstens kans p bevatten | `if total >= top:` |
        | `sorted(p, reverse=True)` | | de getallen van `p` van groot naar klein | `sorted([0.2, 0.5, 0.3], reverse=True)` geeft `[0.5, 0.3, 0.2]` |
        | `np.where(p >= cut, p, 0)` | | een array met elk getal van `p` dat minstens `cut` is, en 0 op de andere plaatsen | `np.where(p >= 0.3, p, 0)` |
        """),
        L.watch_out("Top-k en top-p snijden alleen af; de temperature verandert de vorm van de kansen. Ze worden vaak samen gebruikt: eerst de temperature, dan top-p. Maak k en p niet te klein: met k = 1, of een p die het waarschijnlijkste teken alleen al haalt, is het weer greedy decoding."),
        mo.md(r"""
        De figuur: de waarschijnlijkste tekens na een spatie, van groot naar klein. In kleur staan de tekens die elke methode houdt; de grijze vallen af.
        """),
        mo.Html(C.kept_bars(" ", {"title": "Na een spatie: wat elke methode houdt", "k": "top-k, k = 5: houdt er {n}",
                                  "p": "top-p, p = 0,9: houdt er {n}", "rest": "en nog {n} tekens, allemaal heel onwaarschijnlijk, die beide methoden laten vallen",
                                  "alt": f"Twee rijen van 30 staven, de waarschijnlijkste tekens na een spatie van groot naar klein, te beginnen met t, a, s en h. In de eerste rij, top-k met k = 5, zijn de eerste vijf staven gekleurd. In de tweede rij, top-p met p = 0,9, zijn de eerste {C.kept_after(' ', 0.9)} staven gekleurd. De andere staven zijn grijs."})),
        mo.md(r"""
        Wat het model van de toneelstukken schrijft met seed 1, eerst met gewone sampling, dan met top-p, p = 0,9:
        """),
        mo.Html(C.text_block(C.sample_text(lambda p: p))),
        mo.Html(C.text_block(C.sample_text(lambda p: C.keep_top_p(p, 0.9)))),
        mo.md(f"""
        Geen van beide is Engels: een bigram-model ziet alleen het laatste teken. Maar in de eerste tekst is in {C.tail_pairs(C.sample_text(lambda p: p))} paren het tweede teken er een dat top-p afsnijdt, zoals een regel die met een kleine "r" begint; in de tweede is er geen enkel. De hoofdletters achter elkaar daar zijn geen vreemde trekkingen: de toneelstukken schrijven de namen van de sprekers meestal in hoofdletters.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om alleen de waarschijnlijke tekens van een rij `p` te houden, en daaruit te trekken:

    1. Zoek voor top-k de grens: de k-de grootste kans, `cut = sorted(p)[-k]`.
    2. Houd wat minstens de grens is, en maak de rest 0: `keep = np.where(p >= cut, p, 0)`.
    3. Ga voor top-p de kansen van groot naar klein langs en tel ze op: de kans waarmee het totaal minstens `top` wordt, is de grens. Stop daar met `break`.
    4. Houd wat minstens die grens is, zoals in stap 2.
    5. Deel door de som, `keep / keep.sum()`, en trek met deze kansen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een rij met vier tekens en de kansen 0,2; 0,05; 0,45 en 0,3: top-k met k = 2, dan top-p met p = 0,9. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_cut():
    import numpy as np

    _p = np.array([0.2, 0.05, 0.45, 0.3])   # een rij met vier tekens
    # 1. top-k met k = 2: de grens is de op één na grootste
    _cut = sorted(_p)[-2]
    # 2. houd wat minstens de grens is, 0 voor de rest
    _keep = np.where(_p >= _cut, _p, 0)
    # 5. gedeeld door de som
    print(_keep / _keep.sum())
    # 3. top-p met p = 0.9: van groot naar klein, tot het totaal minstens 0.9 is
    _total = 0
    for _q in sorted(_p, reverse=True):
        _total = _total + _q
        if _total >= 0.9:
            _cut = _q
            break
    # 4. houd wat minstens de grens is
    _keep = np.where(_p >= _cut, _p, 0)
    # 5. gedeeld door de som, en tien trekkingen
    print(_keep / _keep.sum())
    _rng = np.random.default_rng(0)
    print(_rng.choice(len(_p), size=10, p=_keep / _keep.sum()))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Top-k houdt 0,45 en 0,3, en die worden 0,6 en 0,4. Bij top-p gaat het totaal 0,45; 0,75; 0,95: bij 0,2 haalt het 0,9, dus 0,45; 0,3 en 0,2 blijven, en gedeeld door hun som, 0,95, worden ze ongeveer 0,47; 0,32 en 0,21. Teken 1, met 0,05, wordt nooit getrokken. Als meerdere tekens de k-de grootste kans delen, houdt `p >= cut` ze allemaal, dus dan kan top-k er een paar meer dan k houden.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_keeps():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Wat houdt top-p?
    """)
    return


@app.cell(hide_code=True)
def choice_keeps_cell():
    choice_keeps = L.choice(
        "keeps",
        question="Een rij heeft de kansen 0,5; 0,3; 0,15 en 0,05. Welke tekens houdt top-p met p = 0,9?",
        options={
            "first": "Alleen de eerste: 0,5 is de grootste",
            "two": "De eerste twee: 0,5 en 0,3",
            "all": "Alle vier: samen maken ze 1",
            "right": "De eerste drie: samen 0,95",
        },
        correct="right",
        why={
            "first": "0,5 alleen is minder dan 0,9, dus gaat top-p door naar de volgende.",
            "two": "0,5 + 0,3 = 0,8, nog minder dan 0,9.",
            "all": "Top-p stopt zodra het totaal 0,9 haalt: de laatste 0,05 is niet nodig.",
            "right": "0,5 + 0,3 = 0,8 is nog geen 0,9; met 0,15 is het totaal 0,95, dus die drie blijven, en 0,05 wordt afgesneden.",
        },
    )
    choice_keeps
    return (choice_keeps,)


@app.cell(hide_code=True)
def text_exercise_top_k():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Top-k

        `top_k(p, k)` houdt de k waarschijnlijkste kansen van `p`, en deelt ze door hun som. De laatste twee regels staan er al. Vervang de `...` door stap 1 van het recept. De check roept `top_k(np.array([0.1, 0.5, 0.15, 0.2, 0.05]), 2)` aan, en dat moet ongeveer `[0, 0.714, 0, 0.286, 0]` geven; hij probeert ook andere rijen en andere waarden van `k`.
        """),
        L.task("Laat `top_k` als `cut` de k-de grootste kans van `p` nemen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_top_k():
    box_top_k = L.code_box("top-k", lines=8)
    box_top_k
    return (box_top_k,)


@app.cell(hide_code=True)
def check_top_k(box_top_k):
    result_top_k = L.check("top-k", box_top_k.value)
    result_top_k
    return (result_top_k,)


@app.cell(hide_code=True)
def hints_top_k():
    L.hints(
        "top-k",
        [
            "Stap 1 van het recept, zoals in het uitgewerkte voorbeeld.",
            "`sorted(p)` loopt van klein naar groot, dus de grootste staat aan het eind. Welke negatieve index is de k-de vanaf het eind?",
            "`cut = sorted(p)[___]`.",
        ],
        why="Gesorteerd is `[0.1, 0.5, 0.15, 0.2, 0.05]` gelijk aan `[0.05, 0.1, 0.15, 0.2, 0.5]`. De tweede vanaf het eind is 0,2, dus 0,5 en 0,2 blijven; gedeeld door hun som, 0,7, worden ze 0,714 en 0,286.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_top_p():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Top-p

        `top_p(p, top)` houdt de waarschijnlijkste kansen van `p` tot ze samen minstens `top` bevatten; hij heet `top` en niet `p`, omdat `p` al de rij is. De loop staat er al: hij gaat de kansen van groot naar klein langs en telt ze op. Vervang de `...` door de rest van stap 3: een `if` die `q` de grens maakt en de loop stopt. De check roept `top_p(np.array([0.3, 0.1, 0.4, 0.15, 0.05]), 0.8)` aan, en dat moet ongeveer `[0.353, 0, 0.471, 0.176, 0]` geven; hij probeert ook andere rijen en andere waarden van `top`.
        """),
        L.task("Laat `top_p` stoppen bij de kans waarmee het totaal minstens `top` wordt, met die kans als `cut`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_top_p():
    box_top_p = L.code_box("top-p", lines=11)
    box_top_p
    return (box_top_p,)


@app.cell(hide_code=True)
def check_top_p(box_top_p):
    result_top_p = L.check("top-p", box_top_p.value)
    result_top_p
    return (result_top_p,)


@app.cell(hide_code=True)
def hints_top_p():
    L.hints(
        "top-p",
        [
            "Stap 3 van het recept, zoals in het uitgewerkte voorbeeld.",
            "In de loop, nadat het totaal is bijgewerkt: is het al minstens `top`? Dan is deze `q` de grens, en kan de loop stoppen.",
            "`if total >= ___:`, en daarin `cut = ___` en `break`.",
        ],
        why="Van groot naar klein: 0,4; 0,3; 0,15 enzovoort. Het totaal gaat 0,4; 0,7; 0,85: bij 0,15 haalt het 0,8, dus 0,4; 0,3 en 0,15 blijven. Hun som is 0,85, dus ze worden 0,471; 0,353 en 0,176.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De temperature uit [Temperature: gedurfder of veiliger tekst](/course/open/lm-temperature?lang=nl) verandert de vorm van de kansen; top-k en top-p snijden de onwaarschijnlijke af, en de sampling uit [Sampling: laat het model schrijven](/course/open/lm-sampling?lang=nl) trekt uit wat overblijft. Chatdiensten gebruiken vaak allebei: een temperature, en dan top-p. In de volgende les, *Hoe goed is een language model?*, meet je hoe verrast je model is door echte tekst.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_keeps, result_top_k, result_top_p):
    L.wrap_up(choice_keeps, result_top_k, result_top_p)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
