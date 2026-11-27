# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-reproducibility. Route: course/route/lessen/systemen.yaml. Checks, de runs van de course, de demo en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-reproducibility", "nl")
    C = L.checks   # de runs van het cijfer-MLP door de course, de demo en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Willekeurige getallen en seeds](/course/open/np-random?lang=nl) en [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rolls():
    quick_rolls = L.choice(
        "quick-rolls",
        question="Twee random number generators worden allebei gemaakt met `np.random.default_rng(7)`. Elke generator geeft drie dobbelsteenworpen, en daarna geeft de eerste er nog drie. Welke worpen zijn gelijk?",
        options={
            "all": "Alle negen, want beide hebben dezelfde seed",
            "right": "De eerste drie van elke generator",
            "none": "Geen: elke worp is willekeurig",
            "last": "De laatste drie van de eerste en de drie van de tweede",
        },
        correct="right",
        why={
            "right": "Elke generator begint bij de seed, dus geven beide dezelfde eerste drie worpen, en daarna gaat de eerste door met nieuwe worpen. Een training-run krijgt de willekeurige getallen op dezelfde manier: dezelfde seed geeft dezelfde getallen, in dezelfde volgorde.",
            "all": "De eerste generator gaat na drie worpen door, met nieuwe getallen. Welke van deze worpen geeft de tweede generator ook? Zie [Willekeurige getallen en seeds](/course/open/np-random?lang=nl).",
            "none": "De getallen lijken willekeurig, maar volgen uit de seed. Wat geven twee generators met seed 7? Zie [Willekeurige getallen en seeds](/course/open/np-random?lang=nl).",
            "last": "De laatste drie van de eerste generator komen na de eerste drie, dus zijn deze worpen nieuw. Welke worpen zijn gelijk? Zie [Willekeurige getallen en seeds](/course/open/np-random?lang=nl).",
        },
    )
    quick_rolls
    return (quick_rolls,)


@app.cell(hide_code=True)
def quick_check_weights():
    quick_weights = L.choice(
        "quick-weights",
        question="Zonder enige training geeft `DigitsMLP(64)` al tien logits voor een cijfer. Waar komen de weights vandaan?",
        options={
            "zeros": "Allemaal nul, tot de eerste stap",
            "data": "Uit de trainingscijfers, gemiddeld per label",
            "right": "Uit de random number generator van PyTorch",
            "file": "Uit een bestand dat de course bij de les levert, voor elke leerling hetzelfde",
        },
        correct="right",
        why={
            "right": "`nn.Linear` begint met kleine willekeurige getallen als weights, uit de generator van PyTorch. Dus begint het model elke keer hetzelfde als je eerst `torch.manual_seed(0)` aanroept, net als in het project. Deze les voegt de volgorde van de voorbeelden toe, en wat een run verder nodig heeft om twee keer dezelfde uitkomst te geven.",
            "zeros": "Met alleen nullen als weights zou elk hidden neuron hetzelfde berekenen en hetzelfde leren. Waarmee begint `nn.Linear`? Zie [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl).",
            "data": "Zonder training weten de weights nog niets van de cijfers. Waarmee beginnen de weights? Zie [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl).",
            "file": "De course levert de cijfers, niet de weights: `DigitsMLP(64)` maakt de weights zelf. Waarmee? Zie [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl).",
        },
    )
    quick_weights
    return (quick_weights,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In 2018 trainden Peter Henderson en collega's één methode voor reinforcement learning tien keer op dezelfde taak, een gesimuleerde robot die leert rennen, met dezelfde code en dezelfde instellingen. Alleen de seed veranderde. De onderzoekers verdeelden de tien runs in twee groepen van vijf seeds, en een statistische test noemde de twee groepen verschillend, alsof er twee verschillende methoden vergeleken waren. De conclusie: alleen al door andere seeds kan de spreiding tussen runs zo groot zijn dat de uitkomsten statistisch verschillen. Vergelijkt een paper één run van de eigen methode met één run van een andere, dan meet de paper misschien geluk.

    De andere kant is net zo belangrijk. Gaat een run mis, bijvoorbeeld doordat de loss in epoch 7 omhoog springt, dan vind je de oorzaak alleen als je de run opnieuw kunt draaien en dezelfde sprong ziet. Deze les gaat over allebei: een run opnieuw dezelfde uitkomst laten geven, en zien hoe ver runs met andere seeds uit elkaar liggen.

    **Een voorbeeld uit het dagelijks leven.** Eén wedstrijd zegt weinig over wie van twee hardlopers sneller is: op een andere dag, met andere wind, kan de volgorde omdraaien. Een opname van de wedstrijd laat elke keer dezelfde finish zien, maar het blijft één wedstrijd. Om te weten wie sneller is, heb je meerdere wedstrijden nodig.
    """)
    return


@app.cell(hide_code=True)
def predict_spread():
    guess_spread = L.guess(
        "spread",
        question="Je traint het cijfer-MLP uit het project tien keer: dezelfde code, dezelfde data en dezelfde instellingen, met de seeds 0 tot en met 9. Hoe ver liggen de beste en de slechtste run uit elkaar, in goed gelezen testcijfers van de 1.000?",
        options={
            "none": "Helemaal niet: dezelfde code geeft dezelfde uitkomst",
            "two": "Ongeveer 2 cijfers",
            "right": "Ongeveer 15 cijfers",
            "hundred": "Ongeveer 150 cijfers",
        },
        reveal="**Ongeveer 15 cijfers.** Met de seeds 0 tot en met 9 leest de beste run 935 testcijfers goed en de slechtste 919: 16 verschil, met dezelfde code, data en instellingen. Alleen de willekeurige getallen veranderden: de weights aan het begin en de volgorde van de voorbeelden. Draai één seed opnieuw, en je krijgt hetzelfde getal terug. De demo hieronder laat alle tien runs zien.",
    )
    guess_spread
    return (guess_spread,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Waar de willekeurige getallen binnenkomen.** De training van het cijfer-MLP gebruikt op twee plekken willekeurige getallen, allebei uit [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl): de weights van `DigitsMLP` beginnen als kleine willekeurige getallen, en een DataLoader met `shuffle=True` zet de voorbeelden elke epoch in een nieuwe willekeurige volgorde. Beide komen uit de random number generator van PyTorch, en `torch.manual_seed` geeft deze generator een seed, zoals `np.random.default_rng` een seed krijgt in [Willekeurige getallen en seeds](/course/open/np-random?lang=nl). Met dezelfde seed begint een run met dezelfde weights en ziet de run de voorbeelden in dezelfde volgorde.

        **Een eigen generator.** `torch.manual_seed` geeft de globale generator van PyTorch een seed: één generator voor het hele programma, en elk willekeurig getal komt om de beurt uit deze generator. Doe vooraf één willekeurige aanroep extra, zoals een blik op een paar willekeurige voorbeelden, en elk getal daarna schuift op: andere weights, een andere volgorde. Een eigen generator houdt één bron apart. `rng = torch.Generator()` maakt er een, en `rng.manual_seed(1)` geeft de generator een seed. `DataLoader(ds, batch_size=64, shuffle=True, generator=rng)` haalt de volgordes uit deze generator: elke epoch vraagt de DataLoader `torch.randperm` om een willekeurige volgorde van de posities van de voorbeelden, uit deze generator. Met één generator voor de weights en een andere voor de volgorde kun je één seed veranderen en de andere houden.

        **Determinisme.** Met dezelfde seed, dezelfde versies en dezelfde hardware geeft de training van het cijfer-MLP op een CPU bit voor bit hetzelfde model: dat is **determinisme**. Floats maken het lastiger dan het klinkt. Een float bevat een beperkt aantal cijfers, dus moet een optelling vaak afronden, en de volgorde van de optellingen verandert de afronding: het uitgewerkte voorbeeld hieronder telt 0,1, 0,2 en 0,3 in twee volgordes op en krijgt twee verschillende getallen. Een GPU is een chip die duizenden berekeningen tegelijk doet (*Waarom AI GPU's gebruikt, en dezelfde code op de jouwe* legt het uit). Een GPU verdeelt een lange optelling in veel kleine delen, en sommige operaties tellen de delen op in de volgorde waarin de delen klaar zijn. PyTorch gebruikt voor veel GPU-operaties de library cuDNN, en volgens de documentatie van NVIDIA garanderen sommige routines van cuDNN niet dat runs dezelfde uitkomst geven, zelfs niet op hetzelfde soort GPU. Een verschil in het laatste cijfer groeit over duizenden trainingsstappen uit tot een ander model. `torch.use_deterministic_algorithms(True)` laat PyTorch deterministische versies gebruiken, of stoppen met een fout waar er geen is; dat kost vaak snelheid. Zelfs dan waarschuwt de documentatie van PyTorch dat de uitkomsten niet gegarandeerd hetzelfde zijn tussen releases van PyTorch, losse commits of verschillende platforms, en ook niet tussen CPU en GPU, zelfs met dezelfde seeds.

        **Schrijf het op bij de run.** Wil je een run herhalen, dan heb je de seed nodig, de versies (vastgelegd in de lock file uit [Overal dezelfde packages: omgevingen en vaste versies](/course/open/sys-environments?lang=nl)) en de hardware, opgeschreven naast de uitkomst. Samen geven deze gegevens **reproduceerbaarheid**: iemand anders, of jij volgende maand, kan de run opnieuw draaien en dezelfde uitkomst krijgen. De afbeelding laat zien waar een run van afhangt.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **Run-to-run variance.** Ook een herhaalbare run blijft één run. Picard (2021) trainde één klein netwerk voor afbeeldingen met 10.000 verschillende seeds: de test-accuracy liep van 89,01% tot 90,83%, een verschil dat volgens de paper in het vakgebied algemeen als significant geldt. De **run-to-run variance** is hoeveel een uitkomst verschuift als alleen de seed verandert. Om twee instellingen te vergelijken, train je elk met meerdere seeds en kijk je naar de spreiding: de slechtste run, de beste run en het gemiddelde. Eén strenge regel: noem instelling B pas beter dan instelling A als de spreidingen niet overlappen, met alle runs van B boven alle runs van A. Met statistiek kun je met minder runs meer zeggen; deze les houdt het bij de strenge regel.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **reproduceerbaarheid** | | hoe makkelijk jij of iemand anders je werk opnieuw kan draaien en dezelfde uitkomst krijgt | |
        | **determinisme** | | dezelfde code op dezelfde data geeft elke keer precies dezelfde uitkomst | |
        | **run-to-run variance** | | hoeveel een uitkomst verschuift als je hetzelfde experiment opnieuw draait met alleen een andere seed | |
        | **`torch.Generator()`** | | een eigen random number generator, los van de generator die `torch.manual_seed` een seed geeft | `rng = torch.Generator()` |
        | **`torch.randperm(n)`** | rand perm | de getallen 0 tot en met n − 1 in een willekeurige volgorde, als tensor | `torch.randperm(5)` |
        """),
        L.watch_out("In Python is een generator ook een functie die met `yield` waarden één voor één uitdeelt. `torch.Generator` is iets anders: een random number generator."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een run die je kunt herhalen, en een vergelijking die je kunt vertrouwen:

    1. Geef PyTorch aan het begin van de run een seed met `torch.manual_seed(seed)`, voordat je het model maakt.
    2. Geef de volgorde van de voorbeelden een eigen generator met een seed, en geef deze generator met `generator=` aan de DataLoader.
    3. Schrijf de seed, de versies (de lock file) en de hardware op naast de uitkomst.
    4. Moeten runs op een GPU bit voor bit gelijk zijn, voeg dan `torch.use_deterministic_algorithms(True)` toe, en volg de pagina *Reproducibility* van PyTorch voor de instellingen die je GPU nodig heeft.
    5. Vergelijk je twee instellingen, train dan elk met meerdere seeds, zoals vijf of tien, en kijk naar de slechtste run, de beste run en het gemiddelde.
    6. Noem een verandering pas beter als de spreiding boven de oude ligt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Eerst dezelfde drie floats, in twee volgordes opgeteld; dan stap 1 van het recept, twee keer, met `torch.rand(3)`: drie willekeurige floats tussen 0 en 1. `torch.equal` zegt of twee tensors precies dezelfde getallen bevatten. Voer de cel uit.
    """)
    return


@app.cell
def example_seeds():
    import torch

    # floats: dezelfde drie getallen, in twee volgordes opgeteld
    print((0.1 + 0.2) + 0.3)
    print(0.1 + (0.2 + 0.3))

    # 1. een seed: weer dezelfde willekeurige getallen
    torch.manual_seed(0)
    first = torch.rand(3)
    torch.manual_seed(0)
    again = torch.rand(3)
    later = torch.rand(3)
    print(first)
    print(again)
    print(torch.equal(first, again), torch.equal(first, later))
    return again, first, later, torch


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De eerste twee regels verschillen in het laatste cijfer: dezelfde drie getallen, in een andere volgorde opgeteld. Op de CPU van één computer telt PyTorch bij elke run in dezelfde volgorde op; een GPU kan de volgorde per run veranderen. `first` en `again` bevatten dezelfde drie getallen, omdat de generator elke keer eerst opnieuw begon bij seed 0. `later` komt daarna uit dezelfde generator, dus krijgt `later` nieuwe getallen. Haalt een run de willekeurige getallen uit dezelfde seed, in dezelfde volgorde, dan krijgt de run elke keer dezelfde getallen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_orders():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Elke epoch een nieuwe volgorde, elke run dezelfde volgordes

        Een DataLoader met `shuffle=True` en een eigen generator trekt elke epoch een nieuwe volgorde. `orders(n, epochs, seed)` werkt op dezelfde manier voor `n` voorbeelden. `g = torch.Generator()` maakt de generator, en `g.manual_seed(seed)` geeft de generator de seed, één keer, voordat de loop begint. De loop `for epoch in range(epochs)` draait één keer per epoch. Daarin moet `order` een willekeurige volgorde van de getallen 0 tot en met `n` − 1 krijgen van `torch.randperm`, getrokken uit `g`; `order.tolist()` maakt van de tensor een lijst, en `found.append` voegt deze lijst toe aan `found`. Vervang de `...`.

        De check roept `orders(8, 2, 0)` aan, en daarna andere aantallen voorbeelden, epochs en seeds. De afbeelding eronder roept je functie drie keer aan.
        """),
        L.task("Trek de volgorde van elke epoch uit de generator `g`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_orders():
    box_orders = L.code_box("orders", lines=11)
    box_orders
    return (box_orders,)


@app.cell(hide_code=True)
def check_orders(box_orders):
    result_orders = L.check("orders", box_orders.value)
    result_orders
    return (result_orders,)


@app.cell(hide_code=True)
def hints_orders():
    L.hints(
        "orders",
        [
            "Stap 2 van het recept: de volgorde krijgt een eigen generator, `g`. `torch.randperm(n)` geeft de getallen 0 tot en met `n` − 1 in een willekeurige volgorde, als tensor, en `order` bewaart deze tensor.",
            "Met alleen `n` haalt `torch.randperm` de getallen uit de globale generator van PyTorch, en blijft `g` ongebruikt. Welke waarde met een naam geeft `torch.randperm` een generator mee, net als bij een DataLoader?",
            "`        order = torch.randperm(n, ___)`",
        ],
        why="`torch.randperm(n, generator=g)` trekt elke volgorde uit `g`, en `g` gaat van epoch tot epoch door: elke epoch een nieuwe volgorde. Bij de volgende run laat `g.manual_seed(seed)` de generator weer op dezelfde plek beginnen, dus komen de volgordes hetzelfde terug.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Tien seeds, één MLP

    De course trainde het cijfer-MLP uit het project tien keer, met de seeds 0 tot en met 9, en nog tien keer met 128 neuronen in de hidden layer in plaats van 64: elke keer 10 epochs, op een CPU, met PyTorch 2.14.1, de versie in de lock file van de course. De slider kiest een seed. Bovenaan staat de loss van elke epoch voor alle tien seeds met 64 neuronen, met de gekozen seed dik getekend in de kleur van de loss; daaronder de goed gelezen testcijfers van alle twintig runs, met de runs van de gekozen seed groter getekend, in de kleur van de tekst, en de labels waarmee de eerste mini-batch van deze seed begint.
    """)
    return


@app.cell(hide_code=True)
def demo_seed():
    L.explore("seed", "seed", lambda _v: C.frame(_v, L.lang), start=0, low=0, high=9, step=1,
              role="", task="Zoek een seed waarbij 64 neuronen beter zijn dan 128 neuronen, en een seed waarbij 128 neuronen winnen.")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    De loss-curves overlappen bijna, en toch lezen de tien runs met 64 neuronen van 919 tot 935 testcijfers goed. Met 128 neuronen is dat 920 tot 938. Bij vijf seeds winnen 64 neuronen, bij de andere vijf 128 neuronen. De gemiddelden, 927,8 en 929,8, liggen 2 cijfers uit elkaar, veel minder dan de 16 tussen de beste en de slechtste run met 64 neuronen. Met één seed per instelling had je net zo goed het een als het ander kunnen concluderen.

    Waar komt de spreiding vandaan? De course trainde 64 neuronen opnieuw met twee generators. Veranderde alleen de seed van de weights, met de volgorde altijd uit seed 0, dan lag de uitkomst tussen 918 en 931; veranderde alleen de seed van de volgorde, dan tussen 920 en 932. Elke bron geeft alleen al het grootste deel van de spreiding. Een seed voor de ene bron en niet voor de andere geeft je dus niet dezelfde run.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_better():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Beter, of een seed met geluk?

        `better(a, b)` krijgt twee lijsten met de uitkomsten van meerdere seeds: de goed gelezen testcijfers met instelling A, en met instelling B. De functie geeft alleen `True` terug als elke run in `b` elke run in `a` verslaat, en anders `False`. Vervang de `...` door één vergelijking.

        De check roept `better(C.RUNS_64, C.RUNS_128)` aan, met de tien runs uit de demo, en daarna andere lijsten met uitkomsten.
        """),
        L.task("Zeg alleen `True` als elke run van B elke run van A verslaat."),
    ])
    return


@app.cell(hide_code=True)
def exercise_better():
    box_better = L.code_box("better", lines=3)
    box_better
    return (box_better,)


@app.cell(hide_code=True)
def check_better(box_better):
    result_better = L.check("better", box_better.value)
    result_better
    return (result_better,)


@app.cell(hide_code=True)
def hints_better():
    L.hints(
        "better",
        [
            "Elke run van `b` verslaat elke run van `a` als zelfs de slechtste run van `b` dat doet. `min` van Python geeft het kleinste getal in een lijst, dus is `min(b)` deze slechtste run. Welke run van `a` is het moeilijkst te verslaan?",
            "Het moeilijkst te verslaan is de beste run van `a`, het grootste getal in de lijst. Welke ingebouwde functie geeft dat getal?",
            "`    return min(b) > ___`",
        ],
        why="`min(b) > max(a)` is alleen `True` als de slechtste run van `b` de beste run van `a` verslaat: dan verslaat elke run van `b` elke run van `a`. Bij `C.RUNS_64` en `C.RUNS_128` vergelijkt de functie 920 met 935, dus geeft de functie `False`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_gpu():
    mo.md(r"""
    ### Oefening 3. Dezelfde seed op een GPU

    Een collega traint je cijfer-MLP op een GPU, met dezelfde code, dezelfde seed en dezelfde lock file, en leest 929 testcijfers goed in plaats van jouw 932 op een CPU. Een tweede run op dezelfde GPU geeft 930.
    """)
    return


@app.cell(hide_code=True)
def choice_gpu_cell():
    choice_gpu = L.choice(
        "gpu",
        question="Wat is de meest waarschijnlijke oorzaak?",
        options={
            "seed": "De seed werkt op een CPU, niet op een GPU",
            "lock": "De lock file installeerde daar andere versies",
            "right": "De GPU telt getallen op in een wisselende volgorde",
            "test": "De testcijfers komen op een GPU in een andere volgorde, dus verandert het aantal",
        },
        correct="right",
        why={
            "right": "Twee runs op dezelfde GPU verschillen al, dus zit de oorzaak in iets wat van run tot run verandert: de volgorde waarin de GPU getallen optelt. Met `torch.use_deterministic_algorithms(True)` komen de runs op de GPU gelijk uit, vaak ten koste van snelheid; gelijk aan jouw 932 van een CPU hoeven deze runs dan nog niet te zijn.",
            "seed": "`torch.manual_seed` geeft de globale generators van de CPU en van de GPU allebei een seed. En waarom zouden twee runs op dezelfde GPU verschillen?",
            "lock": "De lock file installeert overal dezelfde versies, en twee runs op dezelfde computer verschillen ook. Wat verandert er van de ene run op de andere?",
            "test": "De volgorde van de testcijfers verandert niet hoeveel het model er goed leest. Wat verandert er verder van run tot run?",
        },
    )
    choice_gpu
    return (choice_gpu,)


@app.cell(hide_code=True)
def text_exercise_one_seed():
    mo.md(r"""
    ### Oefening 4. Eén run tegen één run

    Een collega verandert de learning rate van het cijfer-MLP van 0,5 naar 0,4, traint één keer met seed 0, en leest 935 testcijfers goed in plaats van 932.
    """)
    return


@app.cell(hide_code=True)
def choice_one_seed_cell():
    choice_one_seed = L.choice(
        "one-seed",
        question="Wat stel je voor?",
        options={
            "keep": "Houd 0,4: drie cijfers meer goed is beter",
            "right": "Train beide learning rates met elk tien seeds, en vergelijk",
            "again": "Train 0,4 nog een keer met seed 0, voor de zekerheid",
            "pick": "Probeer seeds voor 0,4 tot er een duidelijk boven 932 komt, en meld deze seed",
        },
        correct="right",
        why={
            "right": "Drie cijfers valt ruim binnen de spreiding van tien seeds, dus kan één run per instelling de twee niet onderscheiden. Met tien seeds per instelling zie je of de spreidingen van 0,4 en 0,5 overlappen.",
            "keep": "Met andere seeds lagen de runs van de oude instelling alleen al 16 cijfers uit elkaar. Kunnen drie cijfers van één seed de twee onderscheiden?",
            "again": "Met dezelfde seed en dezelfde versies geeft de run dezelfde uitkomst: weer 935. Wat laat een tweede run met dezelfde seed niet zien?",
            "pick": "Dan meld je de seed met het meeste geluk, en de oude instelling krijgt deze kans niet. Hoe vergelijk je de twee op dezelfde manier?",
        },
    )
    choice_one_seed
    return (choice_one_seed,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Overal dezelfde packages: omgevingen en vaste versies](/course/open/sys-environments?lang=nl) maakte de versies op elke computer gelijk; deze les voegt de seeds, de volgorde van de voorbeelden en de hardware toe. [Willekeurige getallen en seeds](/course/open/np-random?lang=nl) gaf de seed, en [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl) het model dat de course hier vele keren trainde. In de volgende les, *Je runs bijhouden: experiment tracking*, schrijft elke run de seed, de instellingen en de uitkomsten naar een log, zodat je runs later kunt vergelijken.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_gpu, choice_one_seed, result_better, result_orders):
    L.wrap_up(result_orders, result_better, choice_gpu, choice_one_seed)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
