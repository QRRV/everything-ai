# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les stats-mle. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("stats-mle", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Cross-entropy en KL-divergentie](/course/open/info-cross-entropy-kl?lang=nl) en [Hoe goed is een language model?](/course/open/lm-nll?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_kl():
    quick_kl = L.choice(
        "quick-kl",
        question="Is $D_{\\mathrm{KL}}(p \\,\\|\\, q)$ altijd hetzelfde als $D_{\\mathrm{KL}}(q \\,\\|\\, p)$?",
        options={
            "right": "Nee: de waarheid en de verwachting spelen een andere rol",
            "yes": "Ja: het meet hoe ver twee verdelingen uit elkaar liggen",
            "coin": "Ja, bij een munt: met twee uitkomsten maakt de volgorde niet uit",
        },
        correct="right",
        why={
            "yes": "KL is geen afstand: de waarheid weegt de uitkomsten, en de twee omwisselen geeft meestal een ander getal. Zie [Cross-entropy en KL-divergentie](/course/open/info-cross-entropy-kl?lang=nl).",
            "coin": "Ook bij een munt maakt het uit: voor een eerlijke munt $p$ en een verwachte $q$ van 90% kop is $D_{\\mathrm{KL}}(p \\,\\|\\, q)$ ongeveer 0,74 bits, en omgewisseld is $D_{\\mathrm{KL}}(q \\,\\|\\, p)$ ongeveer 0,53. Zie [Cross-entropy en KL-divergentie](/course/open/info-cross-entropy-kl?lang=nl).",
            "right": "In $D_{\\mathrm{KL}}(p \\,\\|\\, q)$ weegt de waarheid $p$ de uitkomsten; wissel ze om en de weging verandert, en het getal ook.",
        },
    )
    quick_kl
    return (quick_kl,)


@app.cell(hide_code=True)
def quick_check_nll():
    quick_nll = L.choice(
        "quick-nll",
        question="Twee language models worden getest op dezelfde echte tekst. De een heeft een gemiddelde NLL van 2,1, de ander 2,4. Welke voorspelt de tekst beter?",
        options={
            "high": "Die met 2,4: dat model is zekerder",
            "right": "Die met 2,1: dat model is minder verrast",
            "same": "Geen van beide: de NLL zegt niets over kwaliteit",
        },
        correct="right",
        why={
            "high": "Een hogere NLL betekent meer verrassing door wat er echt kwam: slechter. Zie [Hoe goed is een language model?](/course/open/lm-nll?lang=nl).",
            "same": "De gemiddelde NLL is precies de maat voor hoe goed een model echte tekst voorspelt. Zie [Hoe goed is een language model?](/course/open/lm-nll?lang=nl).",
            "right": "Lager is beter: het model met 2,1 gaf de echte volgende tekens hogere kansen.",
        },
    )
    quick_nll
    return (quick_nll,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een model heeft parameters, en er komt data binnen. Welke waarden van de parameters kies je? Veel machine learning geeft hetzelfde antwoord: de waarden die de data die je zag het waarschijnlijkst maken. Dat antwoord verklaart waarom language models, classifiers en veel andere modellen allemaal op dezelfde soort loss getraind worden.

    **Een voorbeeld uit het dagelijks leven.** Je vindt een dobbelsteen op straat en gooit er 60 keer mee: 20 zessen, waar een eerlijke dobbelsteen er ongeveer 10 zou geven. Dat hij eerlijk is, geloof je niet meer. Maar in welke kans op zes moet je dan wel geloven, en waarom juist die?
    """)
    return


@app.cell(hide_code=True)
def predict_coin():
    guess_coin = L.guess(
        "seven-of-ten",
        question="Een munt viel 7 van de 10 keer op kop. Je mag zijn kans op kop, $p$, kiezen. Welke $p$ maakt deze 10 worpen volgens jou het waarschijnlijkst?",
        options={
            "fair": "0,5: munten zijn meestal eerlijk",
            "right": "0,7: het aandeel kop",
            "sure": "1: kop kwam het meest",
        },
        reveal="**0,7.** De kans op deze worpen is $p^7 (1 - p)^3$, zeven keer kop en drie keer munt. Bij $p$ = 0,5 is dat ongeveer 0,00098; bij 0,7 ongeveer 0,0022, meer dan twee keer zo waarschijnlijk; bij $p$ = 1 is het 0, want dan had munt nooit kunnen komen. Het aandeel kop is de $p$ die de data het waarschijnlijkst maakt.",
    )
    guess_coin
    return (guess_coin,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        De likelihood uit [Hoe goed is een language model?](/course/open/lm-nll?lang=nl) is de kans die een model geeft aan de data die je zag, als functie van de parameters $\theta$: de kansen $p_\theta(x_i)$ van alle voorbeelden $x_i$, met elkaar vermenigvuldigd. **Maximum likelihood** kiest de $\theta$ die hem het grootst maakt:
        """),
        L.formula(r"\begin{aligned} [[param:\hat{\theta}]] &= \arg\max_\theta \prod_i p_\theta(x_i) \\ &= \arg\min_\theta \; [[loss:-\frac{1}{n} \sum_i \log p_\theta(x_i)]] \end{aligned}"),
        mo.md(r"""
        De twee vormen geven dezelfde $\hat{\theta}$: de log behoudt de volgorde van getallen, dus het grootste product heeft de grootste som van logs; een minteken maakt van het grootste het kleinste, en delen door $n$ verschuift het niet. De tweede vorm is de gemiddelde negative log-likelihood, en dat is de cross-entropy uit [Cross-entropy en KL-divergentie](/course/open/info-cross-entropy-kl?lang=nl) tussen de data en het model. Bij de munt zijn 7 van de 10 termen $-\log p$ en 3 zijn $-\log(1 - p)$, dus het gemiddelde is $-(0{,}7 \log p + 0{,}3 \log(1 - p))$: de cross-entropy met de aandelen in de data als waarheid. Trainen op de cross-entropy-loss is dus maximum likelihood; en omdat de entropie van de data niet verandert, maakt het de KL-divergentie van de data naar het model zo klein als kan.

        Bij een munt is de maximum-likelihoodschatting van $p$ het aandeel kop. Bij een keuze uit meer klassen is de schatting het aandeel van elke klasse: de bigram-tabel uit [De bigram-tabel als matrix van kansen](/course/open/lm-bigram-matrix?lang=nl), elke rij tellingen gedeeld door haar som, is de maximum-likelihoodschatting, en geen andere tabel maakt de tekst waarop hij geteld is waarschijnlijker. Neem je een verwachting over de parameters mee voordat je data ziet, dan komt er een straf bij de loss.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **maximum likelihood** | | de parameters kiezen die de data het waarschijnlijkst maken; hetzelfde als die met de laagste gemiddelde negative log-likelihood kiezen | |
        | $\theta$ | thèta | alle parameters van een model samen; bij de munt alleen $p$ | `p` |
        | $p_\theta(x_i)$ | p thèta van x i | de kans die het model met parameters $\theta$ aan voorbeeld $x_i$ geeft | |
        | $\prod_i$ | product over i | vermenigvuldig de uitdrukking voor elke $i$, zoals $\sum_i$ haar optelt | |
        | $\hat{\theta}$ | thèta-dakje | de schatting van de parameters | `best` |
        | $\arg\max_\theta$, $\arg\min_\theta$ | arg max, arg min over thèta | de $\theta$ waarbij iets het grootst of het kleinst is, niet die waarde zelf | `grid[np.argmin(values)]` |
        | `np.inf` | oneindig | een getal groter dan elk ander getal, een goed begin als je het kleinste zoekt | `np.inf` |
        """),
        L.watch_out("Maximum likelihood past bij de data die je hebt, ook bij de toevalligheden ervan. Een munt die 3 van de 3 keer op kop viel, krijgt $p$ = 1: munt zou onmogelijk zijn. Met weinig data houden een straf of meer data de schatting redelijk."),
        mo.md(r"""
        De figuur: de gemiddelde negative log-likelihood van $p$ voor 7 keer kop in 10 worpen. Ze is het laagst bij het aandeel kop, 0,7.
        """),
        mo.Html(C.coin_picture({"title": "7 keer kop in 10 worpen", "x": "kans op kop, p", "y": "gemiddelde NLL",
                                "best": "het laagst bij 0,7: {v}", "fair": "0,5: {v}", "number": L.number,
                                "alt": "Een kromme van de gemiddelde negative log-likelihood van p, hoog bij 0 en bij 1, met haar laagste punt bij p = 0,7, ongeveer 0,611. Een grijs punt bij p = 0,5, ongeveer 0,693."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om de maximum-likelihoodschatting te vinden:

    1. Schrijf de kans die het model aan één voorbeeld geeft, als functie van de parameter.
    2. De gemiddelde negative log-likelihood: min de log van die kans, gemiddeld over alle voorbeelden.
    3. Zoek de parameter met het laagste gemiddelde: hier door een grid van waarden te proberen.
    4. Vergelijk met het aandeel in de data: bij een munt het aandeel kop.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De munt met 7 keer kop in 10 worpen, als array van enen en nullen, en een grid van waarden van $p$. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 4 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar dat zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_mle():
    import numpy as np

    _data = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1])
    _grid = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    _nlls = []
    for _p in _grid:
        # 1. de kans die p aan elke worp geeft: p voor kop, 1 - p voor munt
        _probs = _data * _p + (1 - _data) * (1 - _p)
        # 2. de gemiddelde negative log-likelihood
        _nlls.append(-np.mean(np.log(_probs)))
    print(np.array(_nlls))
    # 3. de p met het laagste gemiddelde
    _best = _grid[np.argmin(_nlls)]
    print(_best)
    # 4. vergelijk met het aandeel kop in de data: voeg het hier toe, met np.mean
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het laagste van de vijf gemiddelden ligt bij 0,7, en met stap 4 af zie je dat het aandeel kop in de data ook 0,7 is. De gemiddelde NLL daar, ongeveer 0,611, is de entropie van een munt met $p$ = 0,7, in nats: bij de beste $p$ is de cross-entropy van het model gelijk aan de entropie van de data, en is de KL-divergentie 0.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_log():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Waarom de log?
    """)
    return


@app.cell(hide_code=True)
def choice_log_cell():
    choice_log = L.choice(
        "why-log",
        question="Maximum likelihood zoekt het grootste product van kansen. Waarom minimaliseren modellen in plaats daarvan de gemiddelde negative log-likelihood?",
        options={
            "different": "Die geeft een beter antwoord: de log laat kleine kansen zwaarder tellen",
            "faster": "Het product is helemaal niet uit te rekenen, zelfs niet voor een paar voorbeelden",
            "right": "Het beste punt ligt op dezelfde plek, en een som van logs wordt niet afgerond op 0",
        },
        correct="right",
        why={
            "different": "Ze geeft hetzelfde antwoord: de log behoudt de volgorde, dus de beste parameters veranderen niet.",
            "faster": "Voor een paar voorbeelden werkt het product prima; voor miljoenen wordt het in een computer 0, en dat is het probleem.",
            "right": "De log behoudt de volgorde, dus het beste punt blijft waar het was; en een som van miljoenen logs is een gewoon getal, waar het product op 0 zou uitkomen.",
        },
    )
    choice_log
    return (choice_log,)


@app.cell(hide_code=True)
def text_exercise_nll():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De gemiddelde negative log-likelihood

        Schrijf `avg_nll(p, data)`: de gemiddelde negative log-likelihood van `p`, met de natuurlijke log, voor een array `data` van enen en nullen, met stap 1 en 2 van het recept. De check roept `avg_nll(0.75, np.array([1, 0, 1, 1]))` aan, en dat moet ongeveer `0.5623` geven; hij probeert ook andere waarden.
        """),
        L.task("Laat `avg_nll` de gemiddelde verrassing van `data` onder `p` geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_nll():
    box_nll = L.code_box("avg-nll", lines=6)
    box_nll
    return (box_nll,)


@app.cell(hide_code=True)
def check_nll(box_nll):
    result_nll = L.check("avg-nll", box_nll.value)
    result_nll
    return (result_nll,)


@app.cell(hide_code=True)
def hints_nll():
    L.hints(
        "avg-nll",
        [
            "Stap 1 van het recept: een één heeft kans `p`, een nul heeft kans `1 - p`. Vermenigvuldigen met `data` en met `1 - data` kiest voor elk voorbeeld de goede.",
            "Stap 2: min de log van elke kans, en dan het gemiddelde.",
        ],
        why="Een één heeft kans 0,75 en kost $-\\log(0{,}75) \\approx 0{,}288$; de nul heeft 0,25 en kost $-\\log(0{,}25) \\approx 1{,}386$. Drie enen en één nul: $(3 \\cdot 0{,}288 + 1{,}386) / 4 \\approx 0{,}562$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_best():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De waarschijnlijkste p

        Schrijf `best_p(data, grid)`: de `p` uit de array `grid` met de laagste gemiddelde negative log-likelihood voor `data`, met stap 3 van het recept. De check roept `best_p(np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0]), np.array([0.1, 0.3, 0.5, 0.7, 0.9]))` aan, en dat moet `0.3` geven; hij probeert ook andere data en andere grids. Dit vak kent `avg_nll` uit oefening 2 niet: reken het gemiddelde uit in de loop, of kopieer je `avg_nll` in het vak, boven `best_p`.
        """),
        L.task("Laat `best_p` de `p` uit het grid geven die de data het waarschijnlijkst maakt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best():
    box_best = L.code_box("best-p", lines=12)
    box_best
    return (box_best,)


@app.cell(hide_code=True)
def check_best(box_best):
    result_best = L.check("best-p", box_best.value)
    result_best
    return (result_best,)


@app.cell(hide_code=True)
def hints_best():
    L.hints(
        "best-p",
        [
            "Loop over het grid, en reken de gemiddelde negative log-likelihood van elke `p` uit, zoals in oefening 2.",
            "Houd de beste tot nu toe bij: begin met `best_nll = np.inf`, en vervang die, samen met zijn `p`, elke keer dat je een lagere vindt. Of verzamel ze in een lijst en gebruik `np.argmin`.",
        ],
        why="3 enen in 10: het aandeel is 0,3, en dat is de waarde van het grid waar de gemiddelde NLL het laagst is.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De gemiddelde negative log-likelihood uit [Hoe goed is een language model?](/course/open/lm-nll?lang=nl) is de cross-entropy uit [Cross-entropy en KL-divergentie](/course/open/info-cross-entropy-kl?lang=nl), en die zo klein mogelijk maken is maximum likelihood. Daarom trainen zoveel modellen op dezelfde loss. In de volgende les, *Floating point en de log-sum-exp-truc*, zie je wat er gebeurt als een computer die logs en exponenten uitrekent, en hoe je een overflow voorkomt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_log, result_nll, result_best):
    L.wrap_up(choice_log, result_nll, result_best)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
