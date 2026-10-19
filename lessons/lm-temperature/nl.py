# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les lm-temperature. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("lm-temperature", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Sampling: laat het model schrijven](/course/open/lm-sampling?lang=nl) en [Veel klassen: softmax](/course/open/ml-softmax?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_draw():
    quick_draw = L.choice(
        "quick-draw",
        question="`rng.choice(3, p=[0.7, 0.2, 0.1])` draait 1.000 keer. Ongeveer hoe vaak geeft het 2?",
        options={
            "never": "Nooit: 0 is het waarschijnlijkst",
            "right": "Ongeveer 100 keer: één trekking op de tien",
            "third": "Ongeveer 333 keer: elk getal een derde van de keren",
        },
        correct="right",
        why={
            "never": "Dat zou greedy decoding zijn. Sampling trekt ook 2, met kans 0,1. Zie [Sampling: laat het model schrijven](/course/open/lm-sampling?lang=nl).",
            "third": "Dat is trekken zonder `p=`. Met `p=` komt elk getal met zijn eigen kans. Zie [Sampling: laat het model schrijven](/course/open/lm-sampling?lang=nl).",
            "right": "2 heeft kans 0,1, dus het komt ongeveer één keer op de tien: ongeveer 100 van de 1.000.",
        },
    )
    quick_draw
    return (quick_draw,)


@app.cell(hide_code=True)
def quick_check_softmax():
    quick_softmax = L.choice(
        "quick-softmax",
        question="Wat maakt softmax van de logits `[2, 2, 2]`?",
        options={
            "right": "Elk een derde: gelijke logits, gelijke kansen",
            "same": "Elk 2: softmax laat de logits zoals ze zijn",
            "first": "1 voor de eerste, 0 voor de andere",
        },
        correct="right",
        why={
            "same": "Softmax maakt van logits kansen die optellen tot 1, dus drie keer 2 kan de uitkomst niet zijn. Zie [Veel klassen: softmax](/course/open/ml-softmax?lang=nl).",
            "first": "Softmax geeft elke logit een deel; gelijke logits krijgen gelijke delen. Zie [Veel klassen: softmax](/course/open/ml-softmax?lang=nl).",
            "right": "Drie keer e², gedeeld door hun som: elk een derde. Softmax kijkt alleen naar de verschillen tussen de logits, en die zijn er hier niet.",
        },
    )
    quick_softmax
    return (quick_softmax,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je model schrijft door te samplen, en zijn tekst zit vol rare lettercombinaties; greedy decoding blijft hangen in `the the the`. Iets ertussenin zou helpen: trekken, maar de waarschijnlijke tekens meer voortrekken, of minder. Diensten die programma's toegang geven tot een LLM, en apps die er een op je eigen computer draaien, hebben daar vaak precies een instelling voor, de temperature, meestal een getal tussen 0 en 2. In deze les bouw je hem, en zie je wat hij met de tekst van je model doet.

    **Een voorbeeld uit het dagelijks leven.** Een muziek-app op shuffle die je favoriete nummers vaker afspeelt. Een instelling zou hem bijna alleen bij je topnummers kunnen houden, of al je nummers ongeveer even vaak laten afspelen.
    """)
    return


@app.cell(hide_code=True)
def predict_half():
    guess_half = L.guess(
        "half",
        question="In rij \"t\" van je tabel heeft \"h\" de kans 0,34. Je deelt alle logits van die rij door 0,5, waardoor ze verdubbelen, en neemt er opnieuw softmax van. Wat denk je dat er met \"h\" gebeurt?",
        options={
            "same": "Het blijft 0,34: elke logit verandert op dezelfde manier",
            "right": "Het wordt waarschijnlijker: de verschillen verdubbelen",
            "less": "Het wordt minder waarschijnlijk: de andere tekens lopen in",
        },
        reveal=f"**Het wordt waarschijnlijker: {L.number(C.prob_after('t', 'h', 0.5), 2)} in plaats van {L.number(C.prob_after('t', 'h', 1), 2)}.** Als je de logits verdubbelt, verdubbelen ook de verschillen ertussen, en softmax maakt van een groter verschil een groter verschil in kans. De spatie, tweede met {L.number(C.prob_after('t', ' ', 1), 2)}, stijgt naar {L.number(C.prob_after('t', ' ', 0.5), 2)}; alle andere tekens die ooit na \"t\" kwamen, dalen. De logits delen door een getal onder 1 maakt de waarschijnlijke tekens waarschijnlijker.",
    )
    guess_half
    return (guess_half,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een neural network geeft logits, getallen die softmax omzet in kansen, zoals in [Veel klassen: softmax](/course/open/ml-softmax?lang=nl). Je bigram-model geeft meteen de kansen, maar het heeft ook logits: hun logaritmen, `z = np.log(P[i])`. Softmax daarvan geeft `P[i]` terug, want $e$ tot de macht $\log p$ is $p$, en de kansen tellen al op tot 1. Andere getallen met dezelfde verschillen zouden net zo goed werken, want softmax kijkt alleen naar de verschillen; $\log p$ is de eenvoudigste keuze.

        De **temperature** $\tau$ (tau) deelt de logits vóór de softmax. Voor elk van de $K$ tekens is de nieuwe kans $q_k$:
        """),
        L.formula(r"[[pred:q_k]] = \frac{e^{z_k / [[hyper:\tau]]}}{e^{z_0 / [[hyper:\tau]]} + e^{z_1 / [[hyper:\tau]]} + \dots + e^{z_{K-1} / [[hyper:\tau]]}}"),
        mo.md(r"""
        Met $\tau = 1$ verandert er niets. Met $\tau$ onder 1 worden de verschillen tussen de logits groter, dus worden de waarschijnlijke tekens waarschijnlijker en de onwaarschijnlijke onwaarschijnlijker; als $\tau$ dicht bij 0 komt, krijgt het waarschijnlijkste teken een kans van bijna 1. Met $\tau$ boven 1 worden de verschillen kleiner, dus komen de kansen dichter bij elkaar. Een teken dat nooit kwam, blijft bij elke temperature onmogelijk: zijn logit is $\log 0$, min oneindig, en dat gedeeld door $\tau$ is nog steeds min oneindig.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **temperature**, $\tau$ | tau | een getal dat je zelf kiest en dat de logits deelt vóór de softmax: onder 1 scherpere kansen, boven 1 vlakkere | `z / tau` |
        | logits van het bigram-model | | de logaritmen van zijn kansen: softmax daarvan geeft de kansen terug | `np.log(P[i])` |
        """),
        L.watch_out("De naam komt uit de natuurkunde: in een heet gas verdelen de deeltjes zich over veel energieën, en in een koud gas blijven ze dicht bij de laagste. Hier heeft het niets met warmte in de computer te maken: de temperature is gewoon een getal dat je kiest. Door een temperature van 0 kun je niet delen; de meeste programma's doen bij 0 greedy decoding."),
        mo.md(r"""
        De figuur: beweeg de slider. De staven zijn de zes waarschijnlijkste tekens na "t" bij die temperature, en daaronder staat wat het model van de toneelstukken ermee schrijft, beginnend met "T", met seed 1.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_tau_cell():
    def _frame(tau):
        tau = round(tau, 1)
        text = C.sample_text(tau)
        names = {"␣": "de spatie"}   # wat een schermlezer voor ␣ moet zeggen
        bars = ", ".join(f"{names.get(c, c)} {round(p * 100)}%" for c, p in C.after("t", tau, 6))
        return (C.bars("t", tau, {"title": f"Na \"t\", bij τ = {L.number(tau, 1)}", "alt": f"Zes staven, de waarschijnlijkste tekens na t bij τ = {L.number(tau, 1)}, op een vaste schaal waarop een volle staaf kans 1 is: {bars}."})
                + C.text_block(text),
                f"""Van de {len(text) - 1} trekkingen na de "T" zijn er <span class="ea-num">{C.most_likely(text)}</span> het waarschijnlijkste teken na het teken ervoor.""")

    explore_tau = L.explore("tau", "τ", _frame, start=1, low=0.1, high=2.5, step=0.1, role="hyper", task="Wat gaat er mis in de tekst bij $\\tau$ = 0,3, en wat bij $\\tau$ = 2,5?")
    explore_tau
    return


@app.cell(hide_code=True)
def text_after_picture():
    mo.md(r"""
    Bij $\tau$ = 0,3 zijn 115 van de 159 trekkingen het waarschijnlijkste teken, bijna drie op de vier, en valt de tekst steeds terug in "the the": hij is te veilig, en dichter bij 0 wordt sampling greedy decoding. Bij $\tau$ = 2,5 is de tekst te gedurfd: hoofdletters midden in een woord, vraagtekens, hier en daar een apostrof. Daartussen heeft de tekst allebei: vooral waarschijnlijke tekens, en af en toe een minder waarschijnlijk. Welke temperature het best is, hangt af van wat je wilt: voor een antwoord met één goede vorm laag, voor iets nieuws hoger.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om het volgende teken te trekken bij temperature `tau`, uit de kansen `p` in een rij:

    1. Neem de logits van de rij: `z = np.log(p)`.
    2. Deel ze door de temperature: `z = z / tau`.
    3. Softmax, zoals in [Veel klassen: softmax](/course/open/ml-softmax?lang=nl): `e = np.exp(z - z.max())`, en dan `e / e.sum()`.
    4. Trek het volgende teken met deze kansen in plaats van `p`: `rng.choice(len(p), p=...)`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een rij met drie tekens en de kansen 0,5, 0,3 en 0,2, bij de temperatures 0,5, 1 en 2. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_temperature():
    import numpy as np

    _p = np.array([0.5, 0.3, 0.2])   # een rij met drie tekens
    for _tau in [0.5, 1, 2]:
        # 1. de logits
        _z = np.log(_p)
        # 2. gedeeld door de temperature
        _z = _z / _tau
        # 3. softmax
        _e = np.exp(_z - _z.max())
        _q = _e / _e.sum()
        print(_tau, _q)
    # 4. tien trekkingen met de kansen van de laatste temperature, 2
    _rng = np.random.default_rng(0)
    print(_rng.choice(len(_p), size=10, p=_q))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Bij 0,5 gaat het eerste teken van 0,5 naar 0,66 en het laatste van 0,2 naar 0,11: scherper. Bij 1 blijven de kansen zoals ze waren. Bij 2 schuiven ze op naar elk een derde: 0,42, 0,32 en 0,26. In de tien trekkingen bij temperature 2 komt het laatste teken, nummer 2, drie keer.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pick():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Kies een temperature
    """)
    return


@app.cell(hide_code=True)
def choice_pick_cell():
    choice_pick = L.choice(
        "pick-tau",
        question="Je model maakt te veel rare lettercombinaties, maar greedy decoding blijft hangen. Welke temperature probeer je eerst?",
        options={
            "zero": "0: altijd het waarschijnlijkste teken",
            "one": "1: de kansen zoals ze zijn",
            "right": "0,7: de waarschijnlijke tekens wat waarschijnlijker",
            "two": "2: alle tekens die ooit kwamen, dichter bij elkaar",
        },
        correct="right",
        why={
            "zero": "Een temperature van 0 is greedy decoding, en dat blijft hangen.",
            "one": "Bij 1 verandert er niets: de rare combinaties blijven.",
            "two": "Boven 1 wordt de tekst gedurfder: nog meer rare combinaties.",
            "right": "Onder 1 maar niet dicht bij 0: de waarschijnlijke tekens winnen vaker, en het model trekt nog steeds.",
        },
    )
    choice_pick
    return (choice_pick,)


@app.cell(hide_code=True)
def text_exercise_function():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Kansen bij een temperature

        `with_temperature(p, tau)` geeft de kansen `p` bij temperature `tau`. De logits en de laatste regel staan er al. Vervang de `...` door stap 2 en 3 van het recept, in twee regels: eerst de nieuwe `z`, gedeeld door `tau`, dan `e`. De check roept `with_temperature(np.array([0.6, 0.3, 0.1]), 0.5)` aan, en dat moet ongeveer `[0.783, 0.196, 0.022]` geven; hij probeert ook andere temperatures en een kans van 0.
        """),
        L.task("Laat `with_temperature` de logits door `tau` delen en `e` tot de macht van elk nemen, met de grootste er eerst af."),
    ])
    return


@app.cell(hide_code=True)
def exercise_function():
    box_temperature = L.code_box("with-temperature", lines=9)
    box_temperature
    return (box_temperature,)


@app.cell(hide_code=True)
def check_function(box_temperature):
    result_temperature = L.check("with-temperature", box_temperature.value)
    result_temperature
    return (result_temperature,)


@app.cell(hide_code=True)
def hints_function():
    L.hints(
        "with-temperature",
        [
            "Stap 2 en 3 van het recept, zoals in het uitgewerkte voorbeeld.",
            "Eerst een nieuwe `z`: de oude gedeeld door de temperature. Dan `e`: `np.exp` van `z`, met `z.max()` er eerst af.",
            "`z = z / ___` and `e = np.exp(z - ___)`.",
        ],
        why="De logits van `[0.6, 0.3, 0.1]` zijn ongeveer −0,51, −1,20 en −2,30. Gedeeld door 0,5 worden ze −1,02, −2,41 en −4,61: de verschillen zijn twee keer zo groot. Softmax daarvan geeft ongeveer 0,783, 0,196 en 0,022.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_never():
    mo.md(r"""
    ### Oefening 3. Een teken dat nooit kwam
    """)
    return


@app.cell(hide_code=True)
def choice_never_cell():
    choice_never = L.choice(
        "never-seen",
        question="Na \"q\" kwam in de toneelstukken nooit iets anders dan \"u\": zijn rij heeft een 1 voor \"u\" en een 0 voor de andere 64 tekens. Wat maakt een temperature van 2 van die rij?",
        options={
            "right": "Er verandert niets: nog steeds 1 voor \"u\" en 0 voor de rest",
            "spread": "De kansen spreiden zich een beetje: de andere 64 krijgen een kleine kans",
            "half": "\"u\" krijgt 0,5: delen door 2 halveert het",
        },
        correct="right",
        why={
            "spread": "De andere 64 hebben een logit van log 0, min oneindig, en gedeeld door 2 is dat nog steeds min oneindig: hun kans blijft 0. Zie *Het idee*.",
            "half": "De temperature deelt de logits, niet de kansen, en softmax laat ze weer optellen tot 1. Zie *Het idee*.",
            "right": "Een temperature verschuift alleen de kansen van tekens die ooit kwamen. Een teken dat nooit kwam, houdt een logit van min oneindig, dus zijn kans blijft 0, en \"u\" houdt alles.",
        },
    )
    choice_never
    return (choice_never,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Softmax uit [Veel klassen: softmax](/course/open/ml-softmax?lang=nl) maakt van logits kansen; de logits eerst door een temperature delen maakt ze scherper of vlakker, en de sampling uit [Sampling: laat het model schrijven](/course/open/lm-sampling?lang=nl) trekt daar dan uit. Elk language model dat schrijft, tot het grootste LLM, heeft deze instelling. Je bigram-model geeft nog steeds een kans van 0 aan elk paar dat het nooit zag, en geen temperature verandert dat. Een extra les, *Top-k en top-p sampling*, laat nog twee manieren zien om de onwaarschijnlijke tekens buiten te houden. In de volgende les, *Hoe goed is een language model?*, meet je hoe verrast je model is door echte tekst; daarna zie je in *Paren die je nooit zag: smoothing* waarom die nullen een probleem zijn, en los je dat op.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_never, choice_pick, result_temperature):
    L.wrap_up(choice_pick, result_temperature, choice_never)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
