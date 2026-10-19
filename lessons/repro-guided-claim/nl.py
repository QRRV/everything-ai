# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les repro-guided-claim. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("repro-guided-claim", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een paper kiezen die je kunt nabouwen](/course/open/res-choose-paper?lang=nl) en [Overfitting en underfitting](/course/open/ml-overfitting?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_cost():
    quick_cost = L.choice(
        "quick-cost",
        question="Je kent de FLOPs van één run. Waarmee vermenigvuldig je ze, voor de kosten van een hele reproductie?",
        options={
            "params": "Nog een keer met het aantal parameters",
            "right": "Met de runs en de seeds, en een marge voor runs die misgaan",
            "runs": "Alleen met het aantal runs",
        },
        correct="right",
        why={
            "params": "6ND telt de parameters al mee: de FLOPs van één run bevatten ze. Zie [Een paper kiezen die je kunt nabouwen](/course/open/res-choose-paper?lang=nl).",
            "runs": "Elke run heeft meerdere seeds nodig, en sommige runs gaan mis. Zie [Een paper kiezen die je kunt nabouwen](/course/open/res-choose-paper?lang=nl).",
            "right": "Voor de grokking-paper kwam de schatting van de vorige les op ongeveer 12 uur op een laptop: drie seeds, met een marge van 2.",
        },
    )
    quick_cost
    return (quick_cost,)


@app.cell(hide_code=True)
def quick_check_overfit():
    quick_overfit = L.choice(
        "quick-overfit",
        question="Een model heeft 100% van zijn trainingsvoorbeelden goed en 2% van zijn validatievoorbeelden. Wat is er aan de hand?",
        options={
            "right": "Het overfit: het heeft de trainingsvoorbeelden uit het hoofd geleerd",
            "under": "Het underfit: het is te klein om de taak te leren",
            "bug": "De data is stuk: de validation set moet wel verkeerde labels hebben",
        },
        correct="right",
        why={
            "under": "Een model dat te klein is, heeft ook de trainingsvoorbeelden fout. Zie [Overfitting en underfitting](/course/open/ml-overfitting?lang=nl).",
            "bug": "Dat kan, maar een perfecte trainingsscore met een slechte validatiescore is het gewone teken van overfitting. Zie [Overfitting en underfitting](/course/open/ml-overfitting?lang=nl).",
            "right": "Daar begint de paper van deze les: wat als je toch doortraint?",
        },
    )
    quick_overfit
    return (quick_overfit,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Vanaf hier bouw je een paper na, van begin tot eind: *Grokking* van Alethea Power en anderen (2022). Een kleine transformer leert een tabel van sommen, zoals alle $a + b$ modulo 97. Al snel heeft hij elke trainingssom goed en de andere sommen fout: hij overfit. Wat er gebeurt als je toch doortraint, is de claim van de paper, en de claim die jij gaat toetsen. Iedereen doet deze paper samen, in zeven stappen, en elke stap voegt een stuk toe aan je eigen reproductiekit, die je daarna meeneemt naar een paper van je eigen keuze. Deze eerste stap: de paper lezen, de ene claim kiezen die je toetst, en vóór elke run vastleggen wanneer je reproductie mislukt is.
        """),
        draw.project_steps(["Lees de paper en kies de claim", "Bouw het model en de data", "Sanity checks", "Reproduceer de kernfiguur",
                            "De ablation", "Vergelijk met de paper", "Het rapport"],
                           0, here_label="← vandaag", title="Waar je project staat: je reproductie"),
    ])
    return


@app.cell(hide_code=True)
def predict_later():
    guess_later = L.guess(
        "after-overfit",
        question="Een kleine transformer heeft al zijn trainingssommen goed, en bijna geen van de andere. Je traint hem nog veel langer door, met weight decay. Wat denk je dat er met zijn validation accuracy gebeurt?",
        options={
            "stays": "Hij blijft laag: het model heeft de trainingssommen uit het hoofd geleerd",
            "falls": "Hij daalt verder: hoe langer je traint, hoe meer het overfit",
            "right": "Hij stijgt tot bijna 100%: laat, en dan snel",
        },
        reveal="**Hij stijgt tot bijna 100%: laat, en dan snel.** Dat noemt de paper grokking: generalisatie lang na overfitting. Het is de claim die je gaat toetsen, en de paper meldt het bij sommige instellingen wel en bij andere niet, dus een reproductie moet zeggen welke instelling ze toetst.",
    )
    guess_later
    return (guess_later,)


@app.cell(hide_code=True)
def text_reading():
    mo.md(r"""
    ## Leeswijzer

    Download de paper van arXiv: *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*, arXiv:2201.02177. Lees hem met de eerste twee rondes van [Een paper lezen in drie rondes](/course/open/res-three-pass?lang=nl), samen ongeveer een uur, met deze vragen; de derde ronde, elke instelling van het model, hoort bij de volgende stap, als je hem bouwt:

    1. **Het abstract en figuur 1:** wat is grokking, en bij welke taak en welk deel van de data laat figuur 1 het zien?
    2. **Hoe de data gemaakt wordt:** welke vergelijkingen, welke getallen, en hoe ze verdeeld worden over training en validatie.
    3. **De andere figuren:** wat verandert er als het deel training data verandert, en wat doet weight decay? Noteer ze voor latere stappen; deze stap toetst één claim.

    De notatie van de paper naast die van de course en de code:

    | Paper | Course | Code |
    |---|---|---|
    | $a \circ b = c$ | een som uit de tabel, zoals $a + b$ modulo $p$ | `a`, `b`, `c` |
    | $p = 97$ | de getallen 0 tot en met 96 | `p = 97` |
    | training data fraction | het deel van alle sommen voor de training | `train_fraction` |
    | optimization steps | stappen van de training | `steps` |
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Stap 1: één claim.** Een paper doet meerdere claims. Voor een reproductie kies je er één, de kern, die één figuur laat zien en die in je budget past: bij grokking dat de validation accuracy bijna 100% haalt, lang nadat de training accuracy dat deed. De andere claims, over het deel van de data en over weight decay, komen terug in de ablation.

        **Stap 2: meetbaar.** Woorden als "lang daarna" en "bijna 100%" worden getallen: een accuracy telt als gehaald bij 99%, en "lang daarna" betekent op een stap die minstens 10 keer zo laat is. Dan geeft één run twee stapnummers, $t$ voor de training en $v$ voor de validatie, en geldt de claim voor die run als $v \geq 10\,t$.

        **Stap 3: elke uitkomst benoemd.** Een run kan op vier manieren lopen: de training haalt nooit 99% (*did not fit*), de validatie haalt het nooit (*not reached*), de validatie haalt het maar niet 10 keer later (*no delay*), of wel (*grokked*). Elk zegt iets anders over de reproductie.

        **Stap 4: vastgelegd vóór elke run.** Schrijf de drempels, het budget aan stappen en het aantal seeds eerst op, in code, en houd je eraan. Heb je eenmaal krommen gezien, dan is het verleidelijk elke stijging "laat genoeg" te noemen.

        Het plaatje: een schets van de claim, niet de data van de paper. De training accuracy, doorgetrokken, haalt 99% op stap $t$; de validation accuracy, gestreept, op stap $v$, meer dan 10 keer later.
        """),
        mo.Html(C.sketch_picture({"title": "De claim, geschetst", "x": "trainingsstap", "y": "accuracy", "train": "training", "val": "validatie",
                                  "alt": "Een schets op een logaritmische as van trainingsstappen van 10 tot een miljoen. De training accuracy, een doorgetrokken lijn, stijgt van ongeveer 0 naar 100% tussen stap 100 en 1.000 en haalt 99% op stap t, ongeveer 1.000. De validation accuracy, een gestreepte lijn, blijft rond 0 en stijgt laat, en haalt 99% op stap v, ongeveer 170.000. Een dunne lijn markeert 99%."})),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent |
        |---|---|
        | **grokking** | generalisatie lang na overfitting: de validation accuracy stijgt laat, als de training accuracy al lang bijna 100% is |
        | **reproductieplan** | de claim die je toetst, hoe je hem meet, en wanneer hij mislukt is, opgeschreven vóór elke run |
        | $t$, $v$ | de eerste stap waarop de training accuracy, en de validation accuracy, 99% haalt |
        """),
        L.watch_out("Een mislukte reproductie is een resultaat, maar alleen met een plan dat vooraf gemaakt is. Zonder plan kun je elke uitkomst lezen als \"bijna\", en kan niemand, jijzelf ook niet, zeggen wat je gevonden hebt."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo schrijf je een reproductieplan:

    1. Kies de ene claim die je toetst, met de figuur waar hij op rust.
    2. Maak van de woorden getallen: drempels, een budget aan stappen, een aantal seeds.
    3. Benoem elke uitkomst die een run kan hebben, en wat elk over de claim zegt.
    4. Zet het in code vóór elke run, en houd je eraan.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Nieuw is in een log de eerste stap zoeken waarop iets gebeurt: de training accuracy van een verzonnen run, gelogd op vier stappen.
    """)
    return


@app.cell
def example_first():
    _steps = [50, 500, 5000, 50000]
    _train = [0.4, 0.98, 0.99, 1.0]
    for _i in range(len(_steps)):
        if _train[_i] >= 0.99:
            print("de training haalt 99% op stap", _steps[_i])
            break
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De loop gaat het log op volgorde langs en stopt bij de eerste stap waarop de accuracy 0,99 haalt: stap 5.000, waar hij precies 0,99 is. `>=` laat dat meetellen, en `break` stopt bij de eerste, zodat een latere stap hem nooit vervangt. Een log is maar zo fijn als zijn stappen: de run kan 99% ergens tussen stap 501 en 5.000 gehaald hebben.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_claim():
    mo.md(r"""
    ## Jouw beurt

    ### Oefening 1. De claim om te toetsen
    """)
    return


@app.cell(hide_code=True)
def choice_claim_cell():
    choice_claim = L.choice(
        "which-claim",
        question="Drie claims die je uit de grokking-paper kunt halen. Welke toetst deze reproductie als eerste?",
        options={
            "fraction": "Hoe kleiner het deel training data, hoe meer stappen het kost om te generaliseren",
            "right": "Bij één taak en één deel van de data haalt de validation accuracy bijna 100%, lang nadat de training accuracy dat deed",
            "understand": "Neural networks kunnen rekenen gaan begrijpen",
        },
        correct="right",
        why={
            "fraction": "Dat vraagt veel runs bij veel delen: het is een tweede claim, voor de ablation, als de eerste standhoudt.",
            "understand": "\"Begrijpen\" noemt geen resultaat dat je kunt meten; de figuren van de paper meten accuracy over de stappen.",
            "right": "Eén figuur, één opzet, twee stapnummers per run: klein genoeg om te toetsen, en de kern van de paper.",
        },
    )
    choice_claim
    return (choice_claim,)


@app.cell(hide_code=True)
def text_exercise_verdict():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Je plan, in code

        Maak `first_step(steps, accuracy, threshold)` af, de eerste stap waarop de accuracy de drempel haalt, of `None`, en `verdict(steps, train_acc, val_acc)`, dat `"did not fit"`, `"not reached"`, `"no delay"` of `"grokked"` teruggeeft voor één run, zoals in stap 3, met `THRESHOLD` en `DELAY`. De check roept `verdict([10, 100, 1000, 10000, 100000], [0.3, 0.95, 1.0, 1.0, 1.0], [0.0, 0.01, 0.02, 0.4, 1.0])` aan, wat `"grokked"` moet geven, en probeert elk ander soort run. Is het goed, dan wordt het in je project bewaard als `repro/plan.py`, het eerste deel van je reproductiekit.
        """),
        L.task("Maak `first_step` en `verdict` af: de vier uitkomsten van stap 3, met de drempels bovenaan het plan."),
    ])
    return


@app.cell(hide_code=True)
def exercise_verdict():
    box_verdict = L.code_box("verdict", lines=20)
    box_verdict
    return (box_verdict,)


@app.cell(hide_code=True)
def check_verdict(box_verdict):
    result_verdict = L.check("verdict", box_verdict.value)
    result_verdict
    return (result_verdict,)


@app.cell(hide_code=True)
def hints_verdict():
    L.hints(
        "verdict",
        [
            "`first_step` is het uitgewerkte voorbeeld als functie, met `return` in plaats van `print` en `break`, en `None` na de loop. `verdict` vindt `t` en `v` ermee, en kijkt dan naar de training, de validatie en de vertraging, in die volgorde.",
        ],
        why="`first_step`: `for i in range(len(steps)):` met `if accuracy[i] >= threshold: return steps[i]`, en `return None` na de loop. `verdict`: `t = first_step(steps, train_acc, THRESHOLD)` en `v` net zo voor `val_acc`; `\"did not fit\"` als `t is None`, `\"not reached\"` als `v is None`, `\"no delay\"` als `v < DELAY * t`, en anders `\"grokked\"`.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    Je `verdict`, uit je project, op drie verzonnen runs die allemaal op stap 300 99% halen op hun training data. In run A haalt de validation accuracy 99% op stap 30.000, in run B op stap 1.000, kort na de training, en in run C stijgt hij nauwelijks.
    """)
    return


@app.cell(hide_code=True)
def demo_verdict(result_verdict):
    result_verdict  # opnieuw uitvoeren als oefening 2 je plan bewaart
    _result = C.demo()
    if _result is None:
        _out = mo.md("*Maak eerst oefening 2: dan staat je plan in je project, en gebruikt deze cel het.*")
    elif _result == "error":
        _out = mo.md("*Je `repro/plan.py` stopt met een fout. Heb je hem na de check veranderd? Verwijder hem, en check oefening 2 opnieuw.*")
    else:
        _out = mo.md("\n".join(f"- **Run {_name}:** `{_v}`" for _name, _v in zip("ABC", _result)))
    _out
    return


@app.cell(hide_code=True)
def text_exercise_before():
    mo.md(r"""
    ### Oefening 3. Waarom vooraf
    """)
    return


@app.cell(hide_code=True)
def explain_before():
    explain_before_cell = L.explain(
        "why-before",
        question="Waarom schrijf je op wanneer de reproductie mislukt is vóór de eerste run, en niet nadat je de krommen gezien hebt?",
        model_answer="**Krommen lokken verhalen uit:** zie je eenmaal een kromme, dan kun je bijna elke stijging \"laat genoeg\" noemen en elk plateau \"bijna 100%\". Kies je de drempels achteraf, dan passen ze bij wat je kreeg, en kan de reproductie niet meer mislukken. **Een resultaat hoe dan ook:** met een plan dat vooraf vastligt, is een run die niet grokked ook een bevinding, en kan een ander je oordeel met dezelfde regel nagaan. **Dezelfde regel voor elke run:** alle seeds, en later de ablation, worden op dezelfde manier beoordeeld, dus verschillen tussen runs komen van de runs, niet van hoe je ze leest.",
        points=[
            "je zegt dat je na het zien van de krommen geneigd bent de drempels aan te passen aan de resultaten, zodat de claim niet meer kan mislukken",
            "je zegt dat een vast plan een mislukte reproductie tot een resultaat maakt dat anderen kunnen nagaan",
            "je zegt dat elke run en seed met dezelfde regel beoordeeld wordt",
        ],
    )
    explain_before_cell
    return (explain_before_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De keuze van [Een paper kiezen die je kunt nabouwen](/course/open/res-choose-paper?lang=nl), gemaakt voor één paper, en de overfitting van [Overfitting en underfitting](/course/open/ml-overfitting?lang=nl), met een wending. De volgende les, *Begeleide reproductie 2: bouw het model en de data*, maakt de tabel van sommen en past je GPT aan op de paper, instelling voor instelling; latere stappen draaien hem en beoordelen elke run met je `verdict`.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_claim, explain_before_cell, result_verdict):
    L.wrap_up(choice_claim, result_verdict, explain_before_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
