# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson lm-naive-bayes. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("lm-naive-bayes", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [De regel van Bayes](/course/open/prob-bayes?lang=nl) en [Paren die je nooit zag: smoothing](/course/open/lm-smoothing?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_bayes():
    quick_bayes = L.choice(
        "quick-bayes",
        question="1 op de 10 mensen heeft een ziekte. Een test is positief bij 90% van de mensen die ziek zijn, en bij 10% van de gezonde mensen. Iemand test positief. Hoe waarschijnlijk is het dat die ziek is?",
        options={
            "hit": "0,9",
            "right": "0,5",
            "joint": "0,09",
        },
        correct="right",
        why={
            "hit": "Dat is P(positief | ziek), andersom. Zie [De regel van Bayes](/course/open/prob-bayes?lang=nl).",
            "joint": "Dat is P(ziek en positief), nog niet gedeeld door P(positief). Zie [De regel van Bayes](/course/open/prob-bayes?lang=nl).",
            "right": "0,9 · 0,1 = 0,09 is ziek en positief, en 0,1 · 0,9 = 0,09 gezond en positief: de helft van de positieve tests.",
        },
    )
    quick_bayes
    return (quick_bayes,)


@app.cell(hide_code=True)
def quick_check_smooth():
    quick_smooth = L.choice(
        "quick-smooth",
        question="Een rij van een teltabel telt op tot 40, in een tabel van 10 tekens. Welke kans krijgt een paar uit die rij dat nooit voorkwam, met Laplace smoothing, k = 1?",
        options={
            "zero": "0",
            "row": "0,025: 1 op 40",
            "right": "0,02: 1 op 40 + 10",
            "even": "0,1: 1 op 10",
        },
        correct="right",
        why={
            "zero": "Dat is zonder smoothing. Zie [Paren die je nooit zag: smoothing](/course/open/lm-smoothing?lang=nl).",
            "row": "De rijsom groeit ook: elk van de 10 tellingen krijgt er 1 bij. Zie [Paren die je nooit zag: smoothing](/course/open/lm-smoothing?lang=nl).",
            "even": "Dat zou een rij zonder tellingen zijn. Zie [Paren die je nooit zag: smoothing](/course/open/lm-smoothing?lang=nl).",
            "right": "(0 + 1) / (40 + 10) = 1 / 50 = 0,02.",
        },
    )
    quick_smooth
    return (quick_smooth,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [De regel van Bayes](/course/open/prob-bayes?lang=nl) maakte één woord, "free", de kans op spam groter: van 0,2 naar 0,75. Een echt bericht heeft veel woorden, en de regel van Bayes heeft dan P(bericht | spam) nodig: hoe vaak precies dit bericht onder de spam zit. Bijna elk bericht is nieuw, dus zijn telling is 0, net als bij de paren die je nooit zag. Losse woorden tellen lukt wel.

    **Een voorbeeld uit het dagelijks leven.** Je vindt een briefje zonder naam op de koelkast: "laat thuis, pizza in de oven". Je broer schrijft altijd "pizza", je moeder vaak "laat". Elk woord is een aanwijzing, en je weegt de aanwijzingen een voor een, zonder te vragen of ze bij elkaar horen.
    """)
    return


@app.cell(hide_code=True)
def predict_right():
    guess_right = L.guess(
        "how-many",
        question=f"Een filter telt de woorden van {len(C.messages()['train'])} korte berichten die de course schreef, waarvan {C.model()['messages']['spam']} spam, en ziet elk woord als een losse aanwijzing. Hoeveel van {len(C.messages()['test'])} nieuwe berichten denk je dat het goed indeelt?",
        options={
            "coin": "Ongeveer 10",
            "most": "Ongeveer 15",
            "right": "19 of 20",
        },
        reveal=f"**{C.right()} van de {len(C.messages()['test'])}.** Spam heeft eigen woorden, zoals \"free\", \"claim\" en \"cash\", dus woorden tellen is voor de meeste berichten genoeg. Welk bericht het fout indeelt, vind je in het speelblok.",
    )
    guess_right
    return (guess_right,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Hele teksten in klassen indelen, zoals spam en ok, heet **text classification**. Voor een bericht met de woorden $w_1, \dots, w_n$ geeft de regel van Bayes voor elke klasse $c$:
        """),
        L.formula(r"P(c \mid w_1, \dots, w_n) = \frac{P(w_1, \dots, w_n \mid c) \cdot P(c)}{P(w_1, \dots, w_n)}"),
        mo.md(r"""
        De onderkant is voor beide klassen hetzelfde, dus om te kiezen heb je alleen de bovenkant nodig. **Naive Bayes** maakt die telbaar met één aanname: binnen een klasse zijn de woorden onafhankelijk van elkaar. Dat heet **conditionele onafhankelijkheid**, en daarmee is de kans op het hele bericht het product van de kansen op zijn woorden. Dat product is de score van de klasse, en de klasse met de hoogste score wint:
        """),
        L.formula(r"\text{score}(c) = P(c) \cdot P(w_1 \mid c) \cdot P(w_2 \mid c) \cdots P(w_n \mid c)"),
        mo.md(r"""
        De aanname klopt niet, vandaar de naam: in de spam van de course heeft de helft van de berichten "free", maar twee derde van die met "now", dus "now" zegt wel iets over "free". Het filter telt een deel van dezelfde aanwijzing twee keer en is te zeker, en toch kiest het vaak de goede klasse.

        Het filter leest een bericht als een **bag of words**: welke woorden erin staan en hoe vaak, niet hun volgorde. Elke kans komt uit tellingen in de trainingsberichten: P(c) is het aandeel van de berichten van klasse c, en P(w | c) krijgt de Laplace smoothing uit [Paren die je nooit zag: smoothing](/course/open/lm-smoothing?lang=nl), zodat een woord dat een klasse nooit had niet haar hele product 0 maakt. Met $V$ de grootte van de vocabulary, het aantal verschillende woorden in de trainingsberichten:
        """),
        L.formula(r"[[param:P(w \mid c)]] = \frac{(\text{telling van } w \text{ in de berichten van } c) + 1}{(\text{aantal woorden in de berichten van } c) + V}"),
        mo.md(r"""
        Een woord dat in geen enkel trainingsbericht staat, valt weg: geen van beide klassen heeft het gezien. En net als in [Hoe goed is een language model?](/course/open/lm-nll?lang=nl) wordt een product van veel kleine kansen 0 in de computer, dus tel je hun logs op:
        """),
        L.formula(r"[[pred:\log \text{score}(c)]] = [[param:\log P(c)]] + \sum_{i=1}^{n} [[param:\log P(w_i \mid c)]]"),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **text classification** | | hele teksten in klassen indelen, zoals spam en ok | |
        | **naive Bayes** | | een classifier die een tekst de klasse met de hoogste score geeft | `log_score(...)` |
        | **conditionele onafhankelijkheid** | | binnen een klasse zegt het ene woord niets over het andere: P(free en now \| spam) = P(free \| spam) · P(now \| spam) | |
        | **bag of words** | | een tekst als de woorden die erin staan en hoe vaak elk voorkomt, zonder hun volgorde | `text.split()` |

        Het speelblok laat het filter elk van de 20 testberichten indelen en volgt één getal, log score(spam) − log score(ok). Elke <span class="c-param">oranje balk</span> is de duw van één woord, log P(w | spam) − log P(w | ok): een vast getal van het getrainde filter, geen uitkomst zoals de kansen van het bigram-model, dus oranje. De eerste balk is de prior, en de <span class="c-pred">blauwe balk</span> de som: rechts van de gestippelde lijn bij 0 heeft spam de hoogste score. Schuif van bericht naar bericht.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_messages_cell():
    _labels = {"title": "Hoe elk woord de score verschuift", "ok_side": "← ok", "spam_side": "spam →", "prior": "prior",
               "left_out": "weggelaten", "sum": "som", "spam": "spam", "ok": "ok"}

    def _frame(_i):
        _v = C.frame(_i, L.lang)
        _alt = "Bericht {i}, {text}: de score begint bij de prior, {prior}, en elk woord dat het filter kent verschuift hem: {steps}. De som is {sum}, dus het filter zegt {said}."
        _said = "Bericht {i}: *{text}*. Het is {label}, en het filter zegt **{said}**."
        return C.waterfall(_i, _labels | {"alt": _alt.format(**_v)}, L.lang), _said.format(**_v)

    _explore = L.explore("message", "bericht", _frame, start=1, low=1, high=len(C.messages()["test"]), step=1, role="",
                         task="Zoek het bericht dat het filter fout indeelt. Welk woord duwt het bericht het hardst naar spam?")
    _explore
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Bericht 12, "we won free concert tickets text me now", is ok, maar het filter zegt spam: "free" duwt het bericht het hardst, met +2,3, daarna "now" en "text". Alle drie zijn woorden van spam, en het filter telt ze op als drie losse aanwijzingen.

    ### Recept

    Zo deel je een bericht in met naive Bayes:

    1. Tel in de trainingsberichten: per klasse het aantal berichten, hoe vaak elk woord erin voorkomt en hun aantal woorden; en de vocabulary.
    2. Houd de woorden van het nieuwe bericht over die in de vocabulary staan.
    3. Per klasse: begin bij log P(c), en tel voor elk woord log((telling + 1) / (woorden van de klasse + V)) op.
    4. De klasse met de hoogste score wint.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Vijf trainingsberichten, waarvan twee spam, en het nieuwe bericht "free cash now please". `_counts["spam"]` is een dictionary in een dictionary: de tellingen van de woorden van spam. `set(_all_words)` houdt elk woord één keer, en `_w in _vocab` vraagt of een woord in die set staat, net als bij een dictionary. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_naive_bayes():
    import numpy as np

    _train = [("spam", "win cash now"), ("spam", "free cash prize now"),
              ("ok", "lunch now"), ("ok", "call me"), ("ok", "free for lunch")]
    # 1. per klasse: haar berichten, hoe vaak elk woord erin voorkomt, en hun aantal woorden
    _messages = {"spam": 0, "ok": 0}
    _counts = {"spam": {}, "ok": {}}
    _n_words = {"spam": 0, "ok": 0}
    _all_words = []
    for _label, _text in _train:
        _messages[_label] = _messages[_label] + 1
        for _w in _text.split():
            if _w in _counts[_label]:
                _counts[_label][_w] = _counts[_label][_w] + 1
            else:
                _counts[_label][_w] = 1
            _n_words[_label] = _n_words[_label] + 1
            _all_words.append(_w)
    _vocab = set(_all_words)   # de vocabulary: elk verschillend woord
    print(_messages, _n_words, len(_vocab))
    print(_counts["spam"])
    # 2. de woorden van het nieuwe bericht die in de vocabulary staan
    _words = []
    for _w in "free cash now please".split():
        if _w in _vocab:
            _words.append(_w)
    print(_words)
    # 3. per klasse: log P(c), plus log P(w | c) voor elk woord, met Laplace smoothing
    _scores = {}
    for _label in ["spam", "ok"]:
        _score = np.log(_messages[_label] / len(_train))
        for _w in _words:
            if _w in _counts[_label]:
                _n = _counts[_label][_w]
            else:
                _n = 0
            _score = _score + np.log((_n + 1) / (_n_words[_label] + len(_vocab)))
        _scores[_label] = _score
        print(_label, _score)
    # 4. de klasse met de hoogste score
    if _scores["spam"] > _scores["ok"]:
        print("spam")
    else:
        print("ok")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Beide klassen hebben 7 woorden, en de vocabulary heeft er 9: vandaar de 16 onder de breukstreep. "please" staat er niet in, dus blijven er 3 woorden over. Voor spam, met 2 van de 5 berichten: log 0,4 + log(2/16) + log(3/16) + log(3/16), ongeveer −6,34. "cash" kwam bij ok nooit voor en krijgt daar 1/16 in plaats van 0; ok eindigt op ongeveer −7,44, dus het bericht is spam.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_unseen():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Een woord dat spam nooit had
    """)
    return


@app.cell(hide_code=True)
def choice_unseen_cell():
    choice_unseen = L.choice(
        "unseen-word",
        question=f"Het filter van de course telde {C.model()['words']['spam']} woorden in zijn {C.model()['messages']['spam']} spamberichten, en zijn vocabulary heeft {len(C.model()['vocab'])} woorden. \"lunch\" staat erin, maar nooit in spam. Wat is P(lunch | spam)?",
        options={
            "zero": "0: het kwam nooit voor in spam",
            "row": f"1 / {C.model()['words']['spam']}, ongeveer 0,007",
            "right": f"1 / {C.model()['words']['spam'] + len(C.model()['vocab'])}, ongeveer 0,0032",
            "vocab": f"1 / {len(C.model()['vocab'])}, ongeveer 0,0059",
        },
        correct="right",
        why={
            "zero": "Dat is zonder smoothing: \"lunch\" zou de score van spam 0 maken. Zie de formule voor P(w | c) in *Het idee*.",
            "row": "De onderkant groeit ook: de grootte van de vocabulary komt bij de woorden van spam. Zie *Het idee*.",
            "vocab": "De woorden van spam tellen ook mee. Zie de formule voor P(w | c) in *Het idee*.",
            "right": f"(0 + 1) / ({C.model()['words']['spam']} + {len(C.model()['vocab'])}) = 1 / {C.model()['words']['spam'] + len(C.model()['vocab'])}: klein, maar niet 0.",
        },
    )
    choice_unseen
    return (choice_unseen,)


@app.cell(hide_code=True)
def text_exercise_score():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De log score van een klasse

        `log_score(words, prior, counts, n_words, n_vocab)` geeft de log score van één klasse: `words` zijn de woorden van het bericht die in de vocabulary staan, `prior` is P(c), `counts` de dictionary met hoe vaak elk woord in die klasse voorkomt, `n_words` het aantal woorden van de klasse en `n_vocab` de grootte van de vocabulary. Vervang de twee `...` door stap 3 van het recept. De check roept `log_score(["free", "cash", "now"], 0.6, counts, 7, 9)` aan, met de tellingen van ok uit het uitgewerkte voorbeeld, en dat moet ongeveer `-7.442` geven; hij probeert ook andere berichten, waarvan één van 300 woorden.
        """),
        L.task("Laat `log_score` beginnen bij de log van de prior, en voor elk woord de log van P(w | c) optellen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_score():
    box_score = L.code_box("log-score", lines=12)
    box_score
    return (box_score,)


@app.cell(hide_code=True)
def check_score(box_score):
    result_score = L.check("log-score", box_score.value)
    result_score
    return (result_score,)


@app.cell(hide_code=True)
def hints_score():
    L.hints(
        "log-score",
        [
            "Stap 3 van het recept, net als in het uitgewerkte voorbeeld.",
            "De score begint bij de log van de prior. Elk woord telt de log van zijn kans erbij op: 1 bij zijn telling, en de grootte van de vocabulary bij het aantal woorden van de klasse.",
            "`score = np.log(___)`, en in de loop `score = score + np.log((n + 1) / (___))`.",
        ],
        why="Voor ok in het uitgewerkte voorbeeld: log 0,6 is ongeveer −0,511, en \"free\", \"cash\" en \"now\" krijgen 2/16, 1/16 en 2/16. Hun logs erbij opgeteld geven ongeveer −7,442.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_naive():
    mo.md(r"""
    ### Oefening 3. Twee woorden die samen voorkomen
    """)
    return


@app.cell(hide_code=True)
def choice_naive_cell():
    choice_naive = L.choice(
        "naive",
        question="In de spam van de course staan \"free\" en \"now\" vaak in hetzelfde bericht. Een nieuw bericht heeft ze allebei. Wat doet naive Bayes ermee?",
        options={
            "right": "Het telt voor allebei de duw naar spam op, alsof het tweede woord iets nieuws vertelt",
            "once": "Het telt ze als één aanwijzing, omdat ze samen voorkomen",
            "pair": "Het leest ze als één paar, zoals een bigram-model zou doen",
        },
        correct="right",
        why={
            "once": "Naive Bayes kijkt nooit welke woorden samen voorkomen. Zie *Het idee*.",
            "pair": "Dat doet een bigram-model; naive Bayes leest een bag of words, zonder volgorde. Zie *Het idee*.",
            "right": "Dat is de naïeve aanname: de twee aanwijzingen overlappen, dus het filter is te zeker.",
        },
    )
    choice_naive
    return (choice_naive,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [De regel van Bayes](/course/open/prob-bayes?lang=nl) gaf de kans op spam na één woord; naive Bayes doet dat voor alle woorden tegelijk. Zijn kansen komen uit tellingen met de Laplace smoothing uit [Paren die je nooit zag: smoothing](/course/open/lm-smoothing?lang=nl), en hun logs worden opgeteld zoals in [Hoe goed is een language model?](/course/open/lm-nll?lang=nl). Elke klasse is eigenlijk een klein language model van losse woorden, en het filter vraagt welke van de twee, gewogen met haar prior, het bericht waarschijnlijker maakt. In de volgende les van de hoofdroute, *Project: hetzelfde bigram-model, geleerd met gradient descent*, leert je bigram-model zijn tabel in plaats van haar te tellen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_naive, choice_unseen, result_score):
    L.wrap_up(choice_unseen, result_score, choice_naive)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
