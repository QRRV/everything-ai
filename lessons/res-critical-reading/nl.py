# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les res-critical-reading. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-critical-reading", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een paper lezen in drie rondes](/course/open/res-three-pass?lang=nl) en [Eerlijke vergelijkingen](/course/open/res-fair-baselines?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_pass():
    quick_pass = L.choice(
        "quick-second",
        question="Wat moet je na de tweede ronde door een paper kunnen?",
        options={
            "rebuild": "De methode vanaf nul opnieuw bouwen",
            "right": "Iemand de hoofdclaim vertellen en het bewijs ervoor",
            "kind": "Zeggen wat voor paper het is en op welk eerder werk hij voortbouwt",
        },
        correct="right",
        why={
            "rebuild": "Daar is de derde ronde voor. Zie [Een paper lezen in drie rondes](/course/open/res-three-pass?lang=nl).",
            "kind": "Dat geeft de eerste ronde je. Zie [Een paper lezen in drie rondes](/course/open/res-three-pass?lang=nl).",
            "right": "De claim en zijn bewijs: deze les controleert of het ene het andere echt draagt.",
        },
    )
    quick_pass
    return (quick_pass,)


@app.cell(hide_code=True)
def quick_check_fair():
    quick_fair = L.choice(
        "quick-fair",
        question="Een nieuwe methode is een week lang getuned. De baseline is één keer gedraaid, met de instellingen uit een oude paper. De nieuwe methode wint met een klein verschil. Wat laat dat zien?",
        options={
            "wins": "Dat de nieuwe methode beter is: hij won de vergelijking",
            "baseline": "Dat de oude instellingen de beste waren voor de baseline",
            "right": "Niet veel: de baseline is minder getuned",
        },
        correct="right",
        why={
            "wins": "Een week tuning kan meer waard zijn dan een klein verschil. Zie [Eerlijke vergelijkingen](/course/open/res-fair-baselines?lang=nl).",
            "baseline": "Niemand heeft de beste instellingen voor de baseline gezocht, dus dat weten we niet. Zie [Eerlijke vergelijkingen](/course/open/res-fair-baselines?lang=nl).",
            "right": "Deze les zet die controle naast drie andere.",
        },
    )
    quick_fair
    return (quick_fair,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Het abstract van een paper zegt wat de auteurs denken te hebben laten zien. De tabellen en figuren zeggen wat ze wél hebben laten zien. Die twee zijn niet altijd hetzelfde, en het verschil is zelden een leugen: één run die er goed uitzag, een baseline die minder zorg kreeg, een metric die toevallig gunstig uitviel voor de nieuwe methode. Voordat je op een paper voortbouwt, hem citeert of hem gaat nabouwen, leg je de claims naast het bewijs.
    """)
    return


@app.cell(hide_code=True)
def predict_best():
    guess_best = L.guess(
        "best-of-four",
        question="Een nieuwe methode is in werkelijkheid precies even goed als de baseline. De auteurs proberen er 4 varianten van, draaien elk één keer, en melden de beste tegen één run van de baseline. Hoe waarschijnlijk is het dat de beste variant de baseline \"verslaat\"?",
        options={
            "half": "Ongeveer 50%",
            "quarter": "Ongeveer 20%",
            "right": "Ongeveer 80%",
        },
        reveal="**Ongeveer 80%.** Alle 5 runs zijn even goed, dus elk heeft evenveel kans om de beste van de 5 te zijn. De baseline wint alleen als hij toevallig de beste is: 1 keer op 5. Een \"overwinning\" komt dus 4 keer op 5, terwijl niets beter is.",
    )
    guess_best
    return (guess_best,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Stap 1: elke claim naast zijn bewijs.** Schrijf elke claim uit het abstract en de conclusies op, en ernaast de tabel, figuur of het getal dat hem moet dragen. Een claim waar niets naast staat, of iets dat minder zegt, is een **overclaim**.

        **Stap 2: de baselines.** Is er een baseline? Is hij sterk, de beste bekende methode, geen oude zwakke? Kreeg hij evenveel tuning, data en compute als de nieuwe methode?

        **Stap 3: de ruis.** Hoeveel seeds per methode? Is het verschil groter dan de run-to-run variance? Met één run per methode zegt een verschil dat kleiner is dan de spreiding tussen seeds niets.

        **Stap 4: de selectie.** Staat alles erin, of alleen waar de nieuwe methode wint: welke metrics, welke datasets, welke modelgroottes, welke varianten? De beste van $n$ even goede varianten verslaat één run van de baseline in $n / (n + 1)$ van de gevallen.
        """),
        mo.Html(C.chance_picture({"title": "Winnen zonder iets te winnen", "x": "geprobeerde varianten, elk één run", "y": "kans op winst", "lang": L.lang,
                                  "alt": "Een kromme van de kans dat de beste van n even goede varianten één run van de baseline verslaat, voor n van 1 tot 20. Hij stijgt van 0,50 bij 1 variant naar 0,80 bij 4, 0,90 bij 9 en 0,95 bij 19."})),
        mo.md(r"""
        Het plaatje: de kans op een "overwinning" terwijl niets beter is, tegen het aantal geprobeerde varianten. Woorden die vaak verder gaan dan het bewijs: *consequent* (één run?), *significant* (één run per methode, geen interval?), *state of the art* (één baseline?), *in het algemeen* of *voor alle* (één dataset?).

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **overclaim** | een claim die meer zegt dan de experimenten laten zien |
        | **sterke baseline** | de beste bekende methode voor de taak, even zorgvuldig getuned als de nieuwe |
        | **cherry-picking** | alleen de runs, metrics of datasets melden waarop een methode er goed uitziet |
        """),
        L.watch_out("Een overclaim maakt het resultaat niet onwaar, en een paper zonder overclaim is daarom nog niet juist. Kritisch lezen vindt wat het bewijs draagt: vaak een kleinere, ware claim, en dat is de claim om op voort te bouwen of na te bouwen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo leg je de claims van een paper naast het bewijs:

    1. Zet de claims uit het abstract en de conclusies op een rij, en naast elke claim het resultaat dat hem moet dragen.
    2. Controleer de baselines: aanwezig, sterk, en met evenveel moeite getuned.
    3. Controleer de ruis: seeds per methode, en een verschil dat groter is dan de spreiding ertussen.
    4. Controleer de selectie: wat is geprobeerd en niet gemeld. Schrijf dan de claim op die het bewijs wél draagt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Nieuw is de aanpak, toegepast op een hele paper. Een verzonnen abstract: *"FastDrop, een nieuw soort dropout, verhoogt de accuracy van een klein netwerk op de handgeschreven cijfers met 1,2 punt, en generaliseert beter dan elke dropoutmethode."* De tabel: dropout 91,0%, FastDrop 92,2%, elk het gemiddelde van 5 seeds, met een spreiding van ongeveer 0,4 tussen de laagste en de hoogste seed; allebei getuned over dezelfde 6 dropoutkansen.

    1. **Claims en bewijs:** "+1,2 punt op de cijfers" heeft een rij in de tabel. "Beter dan elke dropoutmethode" heeft niets ernaast: er is alleen met gewone dropout vergeleken.
    2. **Baselines:** gewone dropout, op dezelfde manier getuned als FastDrop: eerlijk, maar niet de sterkste dropoutmethode die er is.
    3. **Ruis:** 5 seeds per methode, en een verschil van 1,2 tegen een spreiding van 0,4: de runs van de twee overlappen niet eens, dus dit is duidelijk meer dan ruis.
    4. **Selectie:** één dataset, één netwerk, één metric. "Generaliseert beter" in het algemeen is meer dan één dataset laat zien.

    De claim die het bewijs draagt: *op de handgeschreven cijfers verslaat FastDrop gewone dropout, op dezelfde manier getuned, met ongeveer 1,2 punt.* Kleiner, en waar.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_overclaims():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. Vind de overclaims

        Nog een verzonnen paper. De opzet en de tabel:
        """),
        mo.Html(C.paper_card({"title": "WarmNorm: a warmer normalization for small language models", "invented": "Een paper die voor deze les verzonnen is.",
                              "setup_name": "Opzet.", "setup": "Een GPT van 4 layers, getraind op Tiny Shakespeare. Eén run per methode. Learning rate 0,001, voor WarmNorm gekozen uit vijf waarden op de validation set.",
                              "rows": [("validation loss", "1,62", "1,58"), ("tijd per trainingsstap", "100 ms", "103 ms"),
                                       ("te leren getallen, per plek waar het gebruikt wordt", "2d", "2d + 2")]})),
        mo.md(r"""
        Het abstract doet zes claims. Zet elke claim in zijn groep.
        """),
    ])
    return


@app.cell(hide_code=True)
def exercise_overclaims():
    sort_overclaims = L.sort("overclaims")
    sort_overclaims
    return (sort_overclaims,)


@app.cell(hide_code=True)
def check_overclaims(sort_overclaims):
    result_overclaims = L.check("overclaims", sort_overclaims)
    result_overclaims
    return (result_overclaims,)


@app.cell(hide_code=True)
def hints_overclaims():
    L.hints(
        "overclaims",
        [
            "Gebruik het recept: zoek voor elke zin de rij van de tabel of de regel van de opzet die hem moet dragen. Is die er, en zegt hij evenveel als de zin, ook gezien hoe elke methode getuned is en hoeveel runs er zijn?",
        ],
        why="**Laat de tabel zien:** de losses van één run per methode, 1,58 en 1,62, de 3% langere stap, de 2 extra te leren getallen. **Claimt meer:** *een lagere validation loss dan LayerNorm* (de learning rate is alleen voor WarmNorm gekozen, en er is één run per methode), *de nieuwe state of the art* (er is alleen met LayerNorm vergeleken), *bij elke learning rate* (de vergelijking is bij één learning rate gedraaid).",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fake():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Nepoverwinningen, gesimuleerd

        Controleer het plaatje met een simulatie. Maak `fake_win_rate(n_variants, n_trials, seed)` af: in elk van `n_trials` experimenten krijgen elke variant en de baseline één score uit de normale verdeling met gemiddelde 0 en standaardafwijking 1, `rng.normal(...)`, dus allemaal even goed, en hoger is beter. Geef het aandeel experimenten terug waarin de beste variant hoger scoort dan de baseline. De check roept `fake_win_rate(4, 20000, 0)` aan, wat ongeveer 0,8 moet geven, en probeert 1, 9 en 19 varianten.
        """),
        L.task("Maak `fake_win_rate` af: trek de scores met `rng`, en geef terug hoe vaak de beste variant de baseline verslaat."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fake():
    box_fake = L.code_box("fake-wins", lines=10)
    box_fake
    return (box_fake,)


@app.cell(hide_code=True)
def check_fake(box_fake):
    result_fake = L.check("fake-wins", box_fake.value)
    result_fake
    return (result_fake,)


@app.cell(hide_code=True)
def hints_fake():
    L.hints(
        "fake-wins",
        [
            "Trek een array met één rij per experiment en één kolom per variant, en een array met één baselinescore per experiment. De beste variant van elk experiment is het grootste getal in zijn rij; vergelijken geeft een array van True en False, en het gemiddelde daarvan is het aandeel True.",
        ],
        why="`variants = rng.normal(size=(n_trials, n_variants))`, `baseline = rng.normal(size=n_trials)`, en `return (variants.max(axis=1) > baseline).mean()`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_ask():
    mo.md(r"""
    ### Oefening 3. Waar vraag je om
    """)
    return


@app.cell(hide_code=True)
def choice_ask_cell():
    choice_ask = L.choice(
        "first-ask",
        question="Een paper meldt een validation loss die 0,02 onder die van de baseline ligt, met één run van elk. In je eigen runs van een vergelijkbaar model beweegt de loss ongeveer 0,05 van seed tot seed. Waar vraag je de auteurs eerst om?",
        options={
            "size": "Resultaten voor een groter model, om te zien of de winst groeit met de grootte",
            "right": "Meer seeds voor beide methoden, met hun spreiding",
            "metric": "Een tweede metric naast de loss",
        },
        correct="right",
        why={
            "size": "Een groter model vraagt of het resultaat ook daar geldt. Vraag eerst of er wel een resultaat is: het verschil is kleiner dan de ruis.",
            "metric": "Een andere metric uit dezelfde losse runs heeft hetzelfde probleem: één run per methode kan een verschil van 0,02 niet scheiden van ruis van 0,05.",
            "right": "Met een spreiding van 0,05 tussen seeds kan één run per methode geen verschil van 0,02 laten zien. Meer seeds, en de spreiding, laten zien of er überhaupt een verschil is.",
        },
    )
    choice_ask
    return (choice_ask,)


@app.cell(hide_code=True)
def text_exercise_carried():
    mo.md(r"""
    ### Oefening 4. De claim die het bewijs draagt
    """)
    return


@app.cell(hide_code=True)
def explain_carried():
    explain_carried_cell = L.explain(
        "carried-claim",
        question="Ga terug naar de opzet en de tabel van WarmNorm in Oefening 1. Schrijf in een of twee zinnen de claim op die het bewijs wél draagt, zoals aan het eind van het uitgewerkte voorbeeld, en de twee dingen waar je de auteurs eerst om vraagt.",
        model_answer="**Gedragen:** met een GPT van 4 layers op Tiny Shakespeare eindigde één run van WarmNorm, bij een learning rate die voor WarmNorm gekozen is, op een validation loss van 1,58, en één run van LayerNorm bij dezelfde learning rate op 1,62; een trainingsstap duurt 3% langer, en WarmNorm leert op elke plek 2 getallen meer. Of WarmNorm überhaupt beter is, laat dit niet zien. **Vraag om:** LayerNorm getuned over dezelfde vijf learning rates, en meer seeds voor beide methoden, met hun spreiding.",
        points=[
            "je claim blijft binnen de opzet: één kleine GPT, Tiny Shakespeare, één run per methode",
            "je zegt dat de learning rate alleen voor WarmNorm gekozen is, dus dat een deel van het verschil tuning kan zijn",
            "je vraagt om LayerNorm op dezelfde manier getuned, en om meer seeds voor beide methoden, met hun spreiding",
        ],
    )
    explain_carried_cell
    return (explain_carried_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De baselines van [Eerlijke vergelijkingen](/course/open/res-fair-baselines?lang=nl) en de claims en het bewijs van de tweede ronde in [Een paper lezen in drie rondes](/course/open/res-three-pass?lang=nl), nu als één aanpak voor elke paper. De volgende les, *Een paper kiezen die je kunt nabouwen*, gebruikt hem om een claim te kiezen die het waard is om zelf te toetsen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_ask, explain_carried_cell, result_fake, result_overclaims):
    L.wrap_up(result_overclaims, result_fake, choice_ask, explain_carried_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
