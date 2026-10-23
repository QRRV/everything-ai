# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les prob-bayes. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("prob-bayes", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Voorwaardelijke kans](/course/open/prob-conditional?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rain():
    quick_rain = L.choice(
        "quick-rain",
        question="Van 40 regendagen begonnen er 30 met een bewolkte ochtend. Wat is P(bewolkte ochtend | regen)?",
        options={
            "count": "30: de bewolkte ochtenden",
            "rest": "0,25",
            "right": "0,75",
        },
        correct="right",
        why={
            "count": "Een kans is een aandeel tussen 0 en 1: tel alleen onder de 40 regendagen. Zie [Voorwaardelijke kans](/course/open/prob-conditional?lang=nl).",
            "rest": "Dat is het deel van de regendagen dat zonder wolken begon. Zie [Voorwaardelijke kans](/course/open/prob-conditional?lang=nl).",
            "right": "Van de 40 regendagen begonnen er 30 bewolkt: 30 / 40 = 0,75.",
        },
    )
    quick_rain
    return (quick_rain,)


@app.cell(hide_code=True)
def quick_check_turn():
    quick_turn = L.choice(
        "quick-turn",
        question="Je weet dat P(A | B) = 0,6. Wat weet je van P(B | A)?",
        options={
            "right": "Nog niets: die telt onder andere gevallen",
            "same": "Die is ook 0,6: het gaat om dezelfde A en B",
            "rest": "Die is 0,4: samen maken ze 1",
        },
        correct="right",
        why={
            "same": "P(A | B) telt onder de gevallen met B, P(B | A) onder de gevallen met A: dat zijn andere groepen. Zie het blok *Let op* in [Voorwaardelijke kans](/course/open/prob-conditional?lang=nl).",
            "rest": "0,4 is P(niet A | B): de rest van dezelfde groep. Zie [Voorwaardelijke kans](/course/open/prob-conditional?lang=nl).",
            "right": "De voorwaarde bepaalt onder welke gevallen je telt, dus de ene geeft de andere niet. In deze les ontdek je wat je nog meer nodig hebt om hem om te draaien.",
        },
    )
    quick_turn
    return (quick_turn,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Vaak ken je een kans precies andersom. De maker van een medische test zegt hoe vaak die positief is bij mensen die ziek zijn: P(positief | ziek). Maar wie een positieve test heeft, wil P(ziek | positief) weten, en er is geen tabel met tellingen om andersom te tellen, zoals in [Voorwaardelijke kans](/course/open/prob-conditional?lang=nl): alleen een paar kansen. De regel van Bayes draait ze om.

    **Een voorbeeld uit het dagelijks leven.** Een arts weet hoe een ziekte zich laat zien: de meeste mensen die hem hebben, krijgen koorts. Er komt iemand binnen met koorts, en de arts moet andersom redeneren: hoe waarschijnlijk is de ziekte, nu er koorts is?
    """)
    return


@app.cell(hide_code=True)
def predict_test():
    guess_test = L.guess(
        "test",
        question="Een ziekte is zeldzaam: 1 op de 1000 mensen heeft hem. Een test ervoor is positief bij 99% van de mensen die ziek zijn, en ten onrechte positief bij 1% van de gezonde mensen. Je test positief. Hoe waarschijnlijk denk je dat het is dat je ziek bent?",
        options={
            "high": "Ongeveer 99%: de test heeft 99 van de 100 keer gelijk",
            "half": "Ongeveer 50%",
            "right": "Ongeveer 9%",
        },
        reveal=f"**Ongeveer 9%.** Stel je 100.000 mensen voor. {C.count(C.tested()[0], L.lang)} zijn ziek, en {C.count(C.tested()[1], L.lang)} van hen testen positief. {C.count(C.tested()[2], L.lang)} zijn gezond, en 1% van hen, {C.count(C.tested()[3], L.lang)}, testen ook positief. Van de {C.count(C.tested()[1] + C.tested()[3], L.lang)} positieve tests komen er maar {C.count(C.tested()[1], L.lang)} van mensen die ziek zijn: ongeveer 9%. De ziekte is zo zeldzaam dat er bij de vele gezonde mensen meer vals alarm is dan er goede uitslagen zijn bij de weinige zieken. Het is hetzelfde effect als in oefening 3 van [Voorwaardelijke kans](/course/open/prob-conditional?lang=nl), nu uitgerekend uit kansen in plaats van tellingen.",
    )
    guess_test
    return (guess_test,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Hoe vaak A en B samen gebeuren, kun je op twee manieren uitrekenen: als P(A | B) · P(B), het aandeel van A onder de gevallen met B, keer hoe vaak B gebeurt, of als P(B | A) · P(A). Allebei geven ze P(A en B). Stel ze gelijk en deel door P(B): dat is de **regel van Bayes**.
        """),
        L.formula(r"P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}"),
        mo.md(r"""
        P(A) is de **prior**: hoe waarschijnlijk A is voordat je B weet. P(A | B) is de **posterior**: hoe waarschijnlijk A is nadat je B weet. De regel van Bayes werkt de prior bij met het bewijs B: hij vermenigvuldigt de prior met hoe waarschijnlijk het bewijs is bij A, en deelt door hoe waarschijnlijk het bewijs überhaupt is.

        P(B) weet je vaak niet direct. Tel de twee manieren op waarop B kan gebeuren, met A en zonder A; dat is de **wet van de totale kans**:
        """),
        L.formula(r"P(B) = P(B \mid A) \cdot P(A) + P(B \mid \text{niet } A) \cdot P(\text{niet } A)"),
        mo.md(r"""
        met P(niet A) = 1 − P(A). Voor de test uit de gok: P(positief) = 0,99 · 0,001 + 0,01 · 0,999 ≈ 0,011, en P(ziek | positief) = 0,99 · 0,001 / 0,011 ≈ 0,09.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **regel van Bayes** | | P(A \| B) = P(B \| A) · P(A) / P(B): draait een voorwaardelijke kans om | `p_b_given_a * p_a / p_b` |
        | **prior** | | P(A): hoe waarschijnlijk A is vóór het bewijs | `p_a` |
        | **posterior** | | P(A \| B): hoe waarschijnlijk A is na het bewijs B | `bayes(...)` |
        | **base-rate fallacy** | | de prior vergeten: denken dat een positieve uitslag voor iets zeldzaams bijna zeker is | |
        | **wet van de totale kans** | | P(B) als de som van de manieren waarop B kan gebeuren: met A, en zonder A | `p_b` |
        | P(niet A) | P van niet A | de kans dat A niet gebeurt: 1 − P(A) | `p_not_a` |
        """),
        L.watch_out("De prior vergeten is een veelgemaakte fout, de **base-rate fallacy**: bij een test die 99 van de 100 keer gelijk heeft, lijkt een positieve uitslag 99% zeker. Maar als de ziekte zeldzaam is, zijn de meeste positieve uitslagen vals alarm. De prior, hoe vaak de ziekte voorkomt, hoort in de berekening: bovenaan, en in P(B)."),
        mo.md(r"""
        Het speelblok: 100.000 mensen en de test uit de gok. De bovenste balk is iedereen, de onderste zijn hun positieve tests, en het gevulde deel zijn de mensen die ziek zijn. Schuif met de slider: hij zet hoeveel van de 100.000 er ziek zijn, en de test blijft dezelfde. De aantallen zijn afgerond op hele mensen.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_prior_cell():
    _labels = {"title": "Wie is ziek, voor en na een positieve test?", "everyone": "Alle 100.000 mensen",
               "positive": "Hun {pos} positieve tests", "ill": "ziek: {ill} ({share})", "healthy": "gezond: {healthy}",
               "true_pos": "van zieke mensen: {true_pos} ({posterior})", "false_pos": "van gezonde mensen: {false_pos} ({false_share})",
               "alt": "Twee balken. Van alle 100.000 mensen zijn er {ill} ziek, {share}. Van hun {pos} positieve tests komen er {true_pos} van mensen die ziek zijn, {posterior}, en {false_pos} van gezonde mensen."}

    def _frame(_ill):
        _said = "**{ill}** van de 100.000 mensen zijn ziek, 1 op de {one_in}: van de {pos} positieve tests komen er {true_pos} van mensen die ziek zijn, <span class=\"ea-num\">{posterior}</span>."
        return C.bars(_ill, _labels, L.lang), _said.format(**C.frame(_ill, L.lang))

    _explore = L.explore("prior", "ziek per 100.000", _frame, start=100, values=C.ILL, role="",
                         task="Zoek hoeveel van de 100.000 mensen ziek moeten zijn voordat de helft van de positieve tests van mensen komt die ziek zijn.")
    _explore
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Bij 1000 van de 100.000, 1 op de 100, komt de helft van de positieve tests van mensen die ziek zijn: 990 mensen die ziek zijn testen positief, en 990 gezonde mensen ook. Bij minder zieken zijn er onder de vele gezonde mensen meer valse alarmen dan goede uitslagen; bij meer zieken winnen de goede uitslagen. De test veranderde niet: alleen de prior verschoof de posterior, van ongeveer 1% bij 1 op de 10.000 tot 99% bij 1 op de 2.

    ### Recept

    Om P(B | A) om te draaien tot P(A | B):

    1. Schrijf de prior P(A) op, en hoe waarschijnlijk het bewijs B is met A en zonder A: P(B | A) en P(B | niet A).
    2. Reken P(niet A) = 1 − P(A) uit.
    3. Tel de twee manieren op waarop B kan gebeuren: P(B) = P(B | A) · P(A) + P(B | niet A) · P(niet A).
    4. De regel van Bayes: P(A | B) = P(B | A) · P(A) / P(B).
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een spamfilter: 20% van de mails is spam. Het woord "free" staat in 60% van de spammails, en in 5% van de andere mails. Een nieuwe mail bevat "free". Hoe waarschijnlijk is het dat die mail spam is? Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_bayes():
    # 1. de prior, en hoe waarschijnlijk "free" is met en zonder spam
    _p_spam = 0.2
    _p_free_given_spam = 0.6
    _p_free_given_not_spam = 0.05
    # 2. de mails die geen spam zijn
    _p_not_spam = 1 - _p_spam
    print(_p_not_spam)
    # 3. beide manieren waarop "free" kan voorkomen
    _p_free = _p_free_given_spam * _p_spam + _p_free_given_not_spam * _p_not_spam
    print(_p_free)
    # 4. de regel van Bayes
    _p_spam_given_free = _p_free_given_spam * _p_spam / _p_free
    print(_p_spam_given_free)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Van alle mails is 0,6 · 0,2 = 0,12 spam met "free", en 0,05 · 0,8 = 0,04 andere mail met "free": samen 0,16. Daarvan is 0,12 / 0,16 = 0,75 spam. Het woord "free" bracht de kans op spam van de prior 0,2 naar de posterior 0,75.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_news():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Bewijs dat niets zegt
    """)
    return


@app.cell(hide_code=True)
def choice_news_cell():
    choice_news = L.choice(
        "no-news",
        question="Een test is positief bij 30% van de mensen die ziek zijn, en ook bij 30% van de gezonde mensen. Iemand test positief. Wat doet dat met de kans dat diegene ziek is?",
        options={
            "lower": "Die gaat omlaag: de test heeft meestal ongelijk",
            "right": "Niets: de posterior is de prior",
            "up": "Die gaat omhoog, zoals na elke positieve test",
        },
        correct="right",
        why={
            "lower": "Reken het uit met de regel van Bayes: P(B | A) en P(B) zijn hier allebei 0,3. Zie *Het idee*.",
            "up": "Een positieve test helpt alleen als hij bij zieke mensen vaker positief is dan bij gezonde. Zie *Het idee*.",
            "right": "P(positief) = 0,3 · P(ziek) + 0,3 · P(gezond) = 0,3, dus de regel van Bayes geeft 0,3 · P(ziek) / 0,3 = P(ziek). Een test die bij iedereen even vaak positief is, zegt niets.",
        },
    )
    choice_news
    return (choice_news,)


@app.cell(hide_code=True)
def text_exercise_bayes():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De regel van Bayes als functie

        `bayes(p_a, p_b_given_a, p_b_given_not_a)` geeft P(A | B): uit de prior P(A), en hoe waarschijnlijk B is met A en zonder A. Stap 2 van het recept staat er al. Vervang de twee `...` door stap 3 en 4. De check roept `bayes(0.001, 0.99, 0.01)` aan, de test uit de gok, en dat moet ongeveer `0.0902` geven. Hij probeert ook andere kansen.
        """),
        L.task("Laat `bayes` de twee manieren optellen waarop B kan gebeuren, en dan de kans omdraaien."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bayes():
    box_bayes = L.code_box("bayes", lines=6)
    box_bayes
    return (box_bayes,)


@app.cell(hide_code=True)
def check_bayes(box_bayes):
    result_bayes = L.check("bayes", box_bayes.value)
    result_bayes
    return (result_bayes,)


@app.cell(hide_code=True)
def hints_bayes():
    L.hints(
        "bayes",
        [
            "Stap 3 en 4 van het recept, met de namen uit de functie.",
            "`p_b` is een som van twee producten: B met A, en B zonder A, elk keer hoe vaak zijn geval voorkomt. Daarna geeft de functie P(B | A) keer de prior terug, gedeeld door `p_b`.",
            "`p_b = p_b_given_a * p_a + p_b_given_not_a * ___`, en `return p_b_given_a * ___ / p_b`.",
        ],
        why="Met de test uit de gok: `p_b` = 0,99 · 0,001 + 0,01 · 0,999 = 0,01098, en 0,99 · 0,001 / 0,01098 ≈ 0,0902.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_second():
    mo.md(r"""
    ### Oefening 3. Een tweede test
    """)
    return


@app.cell(hide_code=True)
def choice_second_cell():
    choice_second = L.choice(
        "second-test",
        question="Na de positieve test uit de gok ging de kans dat je ziek bent van 0,001 naar ongeveer 0,09. Je doet dezelfde test nog een keer, los van de eerste, en hij is weer positief. Welke prior gaat er voor de tweede test in de regel van Bayes?",
        options={
            "start": "Weer 0,001: hoe zeldzaam de ziekte is",
            "right": "Ongeveer 0,09: wat de eerste test gaf",
            "test": "0,99: hoe vaak de test de ziekte vindt",
        },
        correct="right",
        why={
            "start": "0,001 was je inschatting vóór elke test; na de eerste weet je meer. Zie wat de prior is in *Het idee*.",
            "test": "0,99 is P(positief | ziek): die gaat ook in de regel van Bayes, maar is niet de prior. Zie *Het idee*.",
            "right": "De posterior van de eerste test is je inschatting nu, dus is hij de prior van de tweede. De regel van Bayes geeft dan 0,99 · 0,09 / (0,99 · 0,09 + 0,01 · 0,91) ≈ 0,91: twee positieve tests maken de ziekte waarschijnlijk.",
        },
    )
    choice_second
    return (choice_second,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Voorwaardelijke kans](/course/open/prob-conditional?lang=nl) liet zien dat P(A | B) niet P(B | A) is; de regel van Bayes maakt van de een de ander, met de prior P(A) en de wet van de totale kans voor P(B). Een inschatting die met bewijs is bijgewerkt, wordt de prior voor het volgende bewijs. Klassieke spamfilters werken zo, woord voor woord: met *Toon verdieping* aan bouw je er zelf een in *Spam of niet: naive Bayes*, na *Paren die je nooit zag: smoothing*. In de volgende les, *Soorten leren*, onderscheid je de soorten leren aan de hand van waar de goede antwoorden vandaan komen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_news, choice_second, result_bayes):
    L.wrap_up(choice_news, result_bayes, choice_second)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
