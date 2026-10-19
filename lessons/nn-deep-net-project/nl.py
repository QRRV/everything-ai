# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nn-deep-net-project. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow residual, initialization

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-deep-net-project", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Residual connections](/course/open/nn-residual?lang=nl) en [Batch normalization](/course/open/nn-batchnorm?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_residual():
    quick_residual = L.choice(
        "quick-residual",
        question="Een residual block geeft `x + f(x)`. Stel dat de layers van `f` een gradient van bijna 0 teruggeven. Wat komt er op de terugweg bij `x` aan?",
        options={
            "zero": "Ook bijna 0: alles moet door de layers van `f` heen",
            "right": "De hele gradient van boven: de + geeft hem ongewijzigd door",
            "twice": "Twee keer de gradient van boven, één kopie per pad",
        },
        correct="right",
        why={
            "zero": "De `x` in `x + f(x)` is een tweede pad dat `f` overslaat, en de gradient neemt dat ook. Zie [Residual connections](/course/open/nn-residual?lang=nl).",
            "twice": "De twee paden tellen op: dat langs `f` geeft de gradient één keer, dat door `f` bijna niets. Zie [Residual connections](/course/open/nn-residual?lang=nl).",
            "right": "De afgeleide van `x + f(x)` is 1 plus die van `f`, dus de gradient van boven komt altijd door. Vandaag zijn residual connections een van de oplossingen voor een netwerk dat niet traint.",
        },
    )
    quick_residual
    return (quick_residual,)


@app.cell(hide_code=True)
def quick_check_norm():
    quick_norm = L.choice(
        "quick-norm",
        question="Waar komt `nn.BatchNorm1d` in een hidden layer van een MLP?",
        options={
            "before": "Vóór de `nn.Linear`, op de input van de layer",
            "right": "Tussen de `nn.Linear` en de ReLU, op zijn $z$'s",
            "after": "Na de ReLU, op wat de layer doorgeeft",
        },
        correct="right",
        why={
            "before": "Hij standaardiseert wat de `nn.Linear` geeft, de $z$ van elk neuron. Zie [Batch normalization](/course/open/nn-batchnorm?lang=nl).",
            "after": "Hij standaardiseert de $z$'s voordat de ReLU de negatieve op 0 zet. Zie [Batch normalization](/course/open/nn-batchnorm?lang=nl).",
            "right": "Linear, dan batch normalization, dan ReLU: in het project van vandaag komt hij in elke hidden layer van een diep netwerk.",
        },
    )
    quick_norm
    return (quick_norm,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Diepte maakt netwerken sterk: elke layer kan voortbouwen op wat de layer ervoor vond. Maar hoe dieper het netwerk, hoe meer er mis kan gaan bij het trainen, vaak zonder foutmelding. In echte projecten is een netwerk dat niet goed traint een van de meest voorkomende problemen, en er is een vaste manier om te vinden wat er mis is. Dit project traint een MLP van 10 layers op een nieuwe dataset, **Fashion-MNIST**: 70.000 kleine grijze plaatjes van kleding, schoenen en tassen, 28 bij 28 pixels, in 10 soorten, zoals broeken, sneakers en enkellaarzen, in 2017 gemaakt door de webwinkel Zalando als moeilijkere vervanging voor handgeschreven cijfers. De course gebruikt er 6.000 om te trainen en 1.000 om te testen.
    """)
    return


@app.cell(hide_code=True)
def predict_plain():
    guess_plain = L.guess(
        "plain-deep",
        question="Een MLP met één hidden layer van 64 neuronen leest na 3 epochs SGD met een learning rate van 0,1 tussen de 724 en 807 van de 1.000 testplaatjes goed, afhankelijk van de seed. Nu dezelfde training voor een MLP met 10 layers: 64 neuronen in elke hidden layer, ReLU, en de startgewichten van `nn.Linear` zelf. Hoeveel leest dat er goed?",
        options={
            "more": "Ongeveer 850: meer layers kunnen meer leren",
            "less": "Ongeveer 750: iets slechter, want dieper traint moeilijker",
            "right": "Ongeveer 100: niet beter dan gokken",
        },
        reveal="**105, voor elk van 5 seeds.** Na de training zegt het bij elk plaatje \"broek\", en 105 van de 1.000 testplaatjes zijn broeken. De loss blijft op 2,30, de loss van gokken uit 10 soorten: $\\ln 10 \\approx 2{,}30$. Layer na layer vervagen de verschillen tussen de plaatjes: na de achtste hidden layer zijn ze 700 tot 1.000 keer kleiner dan na de eerste, over 5 seeds, dus elk plaatje geeft dezelfde output.",
    )
    guess_plain
    return (guess_plain,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Traint een netwerk niet, zoek dan eerst uit of het probleem in het netwerk en zijn trainingsstap zit, of ergens anders, zoals in de data. **Overfit één batch**: train het netwerk op één mini-batch, bij elke stap dezelfde 64 voorbeelden, een paar honderd stappen lang. Een netwerk dat werkt, leert die 64 uit het hoofd, en zijn loss gaat naar bijna 0. Lukt zelfs dat niet, dan helpen meer data of meer epochs niet: het probleem zit in het netwerk, zijn start, of de trainingsstap. De test kost seconden, dus hij komt eerst.

        De 10 layers zakken voor de test: na 300 stappen SGD met 0,05 op 64 plaatjes is de loss nog 2,23. De oorzaak is de spreiding van de startgewichten, uit [Initialisatie](/course/open/nn-initialization?lang=nl): die van `nn.Linear` zijn te klein voor veel ReLU-layers, dus het signaal wordt layer na layer kleiner, en de gradient op de terugweg ook. Drie oplossingen uit de vorige lessen werken daar elk tegenin. He initialization geeft de weights meteen de goede spreiding. Batch normalization standaardiseert de $z$'s van elke hidden layer over de mini-batch. Residual connections laten het signaal, en de gradient, elke layer heel passeren. Dan een betere optimizer: AdamW met warmup en cosine decay, je Trainer uit [Een herbruikbare training loop](/course/open/nn-trainer?lang=nl). Verander één ding tegelijk, en meet na elke verandering: dan weet je wat geholpen heeft.

        Een diep netwerk heeft veel layers, en een gewone Python-lijst verstopt layers voor PyTorch: `model.parameters()` vindt alleen de layers die attributen van het model zijn. `nn.ModuleList` is een lijst layers die PyTorch wel ziet. Je voegt een layer toe met `.append`, en je indexeert hem en loopt erover zoals over een lijst.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **overfit één batch** | | train op één mini-batch, om te testen of het netwerk en zijn trainingsstap überhaupt kunnen leren | `overfit_one_batch(model, xb, yb, 300, 0.05)` |
        | `nn.ModuleList` | n n module list | een lijst layers die PyTorch ziet, zodat hun parameters getraind worden | `nn.ModuleList()`, dan `.append(nn.Linear(64, 64))` |
        """),
        L.watch_out("Eén batch overfitten is een test, geen manier om te trainen: een netwerk dat 64 plaatjes uit het hoofd kent, leest nieuwe plaatjes slecht. En slagen voor de test betekent niet dat het netwerk het goed doet op alle data: hij sluit alleen de grofste problemen uit. Verander tijdens het zoeken naar de oorzaak één ding tegelijk; met twee veranderingen tegelijk weet je niet welke hielp."),
        mo.md(r"""
        De figuur: dezelfde 10 layers, 3 epochs getraind op de 6.000 plaatjes, op vijf manieren; elke balk is één run, met seed 0. Over 5 seeds variëren de getallen: He initialization leest er 689 tot 731 goed, batch normalization 748 tot 788, residual connections 718 tot 824, en alle drie samen met AdamW 838 tot 850, meer dan het MLP met één hidden layer. AdamW alleen, zonder de drie oplossingen, haalt 624 tot 659: Adam schaalt de piepkleine gradients op, dus dit netwerk leert wel, maar het blijft achter bij elk van de drie oplossingen. Met de Trainer is één oplossing al genoeg: residual connections alleen lezen er dan 841 tot 851 goed, evenveel als alle drie, en het MLP met één hidden layer haalt 816 tot 824, dus de 10 layers winnen, maar met weinig. En niet elke combinatie helpt: He initialization met residual connections maar zonder batch normalization laat het signaal bij elk blok groeien, en met SGD op 0,1 eindigt de loss als `nan`, voor elk van 5 seeds.
        """),
        mo.Html(C.fixes_picture({"title": "Testplaatjes goed gelezen, van de 1.000", "plain": "startgewichten van nn.Linear, SGD", "he": "He initialization, SGD",
                                 "norm": "batch normalization, SGD", "residual": "residual connections, SGD", "all": "alle drie, met de Trainer: AdamW",
                                 "note": "3 epochs op 6.000 plaatjes; elk één run, seed 0",
                                 "alt": "Vijf balken met de testplaatjes die goed gelezen worden, van de 1.000: met de startgewichten van nn.Linear en SGD 105, met He initialization 690, met batch normalization 787, met residual connections 718, en met alle drie en de Trainer met AdamW 850."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een diep netwerk dat niet traint, stap voor stap:

    1. Overfit één batch. Gaat de loss niet naar bijna 0, kijk dan eerst naar het netwerk en de trainingsstap.
    2. Geef de weights de goede start: He initialization voor ReLU, bias terms op 0.
    3. Voeg batch normalization toe aan elke hidden layer, tussen de `nn.Linear` en de ReLU.
    4. Voeg residual connections toe rond de hidden layers.
    5. Train met AdamW, warmup en cosine decay: je Trainer.

    Verander één ding tegelijk, en meet na elke verandering.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 1 voor twee netwerken: de 10 layers met de startgewichten van `nn.Linear`, en met He initialization, elk 300 stappen getraind op de eerste 64 trainingsplaatjes. `C.DeepMLP` is de versie van de course van de class die je in oefening 3 schrijft. Het commentaar met `#` zegt welke stap van het recept elk deel is. De trainingsstap is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_overfit():
    import torch
    import torch.nn.functional as F

    _xb, _yb = C.one_batch()
    # 2. zonder en met He initialization
    for _he in [False, True]:
        torch.manual_seed(0)
        _model = C.DeepMLP(10, 64, he=_he)
        _opt = torch.optim.SGD(_model.parameters(), lr=0.05)
        # 1. overfit één batch: 300 stappen op dezelfde 64 plaatjes
        for _step in range(300):
            _loss = F.cross_entropy(_model(_xb), _yb)
            _opt.zero_grad()
            # maak af: backward, en de stap van _opt
        print(_he, round(_loss.item(), 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Zoals hij is, zet de loop nooit een stap: beide losses zijn de loss bij de start, 2,3172 voor het gewone netwerk en 2,481 met He initialization. Ook dat is de test die zakt, om een andere reden, die hij goed opspoort: een trainingsstap die ontbreekt. Afgemaakt, met `_loss.backward()` en `_opt.step()`, eindigt het gewone netwerk op 2,2263, vlak bij waar het begon, en dat met He initialization vlak bij 0: het kent de 64 plaatjes uit het hoofd. De course mat 0,0307; je computer kan daar een ander klein getal geven, zoals 0,0019, omdat het kleinste afrondingsverschil over 300 stappen groeit. Over 5 seeds eindigt het gewone netwerk altijd tussen 2,225 en 2,227, en dat met He initialization tussen 0,002 en 0,031.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_overfit_first():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Wat de test je vertelt
    """)
    return


@app.cell(hide_code=True)
def choice_overfit_first_cell():
    choice_overfit_first = L.choice(
        "overfit-first",
        question="Je netwerk kan één mini-batch niet uit het hoofd leren: na 300 stappen op dezelfde 64 plaatjes is de loss nog 2,23. Wat vertelt je dat?",
        options={
            "epochs": "Het heeft meer epochs nodig: 300 stappen zijn te weinig voor 10 layers",
            "data": "De plaatjes zijn te moeilijk: het heeft er meer nodig om van te leren",
            "right": "Het probleem zit in het netwerk, zijn start of de trainingsstap: niet in de data",
        },
        correct="right",
        why={
            "data": "Meer plaatjes helpen niet bij een netwerk dat er zelfs 64 niet uit het hoofd kan leren: die 64 waren alles wat het zag.",
            "epochs": "300 stappen zijn ruim genoeg voor 64 plaatjes: met He initialization komt hetzelfde netwerk in die 300 stappen op 0,03 of lager.",
            "right": "Eén mini-batch is de makkelijkste taak die er is. Een netwerk dat dat niet kan, heeft zelf een probleem, zoals weights die te klein beginnen of een trainingsstap die ontbreekt, dus daar kijk je eerst.",
        },
    )
    choice_overfit_first
    return (choice_overfit_first,)


@app.cell(hide_code=True)
def text_exercise_overfit():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De overfit-test

        Maak `overfit_one_batch(model, xb, yb, steps, lr)` af: die traint `model` `steps` stappen lang met SGD, learning rate `lr`,, elke stap op dezelfde `xb` en `yb`, en geeft de loss van de laatste stap terug. Vervang de `...` door de vier regels van een trainingsstap, zoals in het afgemaakte uitgewerkte voorbeeld. De check traint kleine modellen op willekeurige data en vergelijkt de laatste loss en de weights. Als hij goed is, slaat de check hem op in je project, als `work/projects/nn/overfit.py`.
        """),
        L.task("Laat `overfit_one_batch` een model trainen op één mini-batch en de laatste loss geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_overfit():
    box_overfit = L.code_box("overfit", lines=12)
    box_overfit
    return (box_overfit,)


@app.cell(hide_code=True)
def check_overfit(box_overfit):
    result_overfit = L.check("overfit", box_overfit.value)
    result_overfit
    return (result_overfit,)


@app.cell(hide_code=True)
def hints_overfit():
    L.hints(
        "overfit",
        [
            "Vier regels, elk ingesprongen onder de `for`: de loss van het model op `xb` en `yb`, dan de gradients op 0, dan backward, dan de stap.",
            "De loss is `F.cross_entropy` van de output van het model en `yb`. `return loss.item()` na de loop geeft de loss van de laatste ronde.",
        ],
        why="De loss van elke ronde wordt berekend vóór zijn stap, dus de loss die terugkomt is die van de laatste stap, van vóór zijn update. `opt.zero_grad()` vóór `backward()` zorgt dat de gradients van een ronde niet optellen bij die ervoor.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_deep():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Eén class voor elke versie

        Maak de class `DeepMLP` af: `DeepMLP(depth, width, he, norm, residual)` heeft in totaal `depth` layers, een input layer, `depth - 2` hidden layers van `width` neuronen, en een output layer van 10 logits. `__init__` maakt de layers, de hidden layers in een `nn.ModuleList`, met voor elk een batch normalization als `norm` `True` is. Vervang de twee `...`:

        - In `__init__`, als `he` `True` is: He initialization voor elke `nn.Linear`, ook `self.inp` en `self.out`, met de bias terms op 0, zoals in [Initialisatie](/course/open/nn-initialization?lang=nl).
        - In `forward` de stappen van hidden layer `k`: de layer, zijn batch normalization als die er is, ReLU; dan met `residual` het resultaat bij `h` optellen, en zonder `residual` `h` door het resultaat vervangen.

        De check maakt kleine netwerken na `torch.manual_seed(0)` en vergelijkt wat ze geven voor 3 willekeurige plaatjes, in elke versie, en de spreiding van de weights met He initialization. Dat zijn willekeurige getallen, dus neem de layers in dezelfde volgorde als de course: `self.inp`, dan de hidden layers, dan `self.out`. Als hij goed is, slaat de check hem op als `work/projects/nn/deep_mlp.py`.
        """),
        L.task("Laat `DeepMLP` een diep MLP bouwen met He initialization, batch normalization en residual connections als keuze."),
    ])
    return


@app.cell(hide_code=True)
def exercise_deep():
    box_deep = L.code_box("deep-mlp", lines=34)
    box_deep
    return (box_deep,)


@app.cell(hide_code=True)
def check_deep(box_deep):
    result_deep = L.check("deep-mlp", box_deep.value)
    result_deep
    return (result_deep,)


@app.cell(hide_code=True)
def hints_deep():
    L.hints(
        "deep-mlp",
        [
            "Voor He initialization: een `for`-loop over een lijst van alle `nn.Linear`s, gemaakt met `[self.inp] + list(self.hidden) + [self.out]`, met de twee regels uit [Initialisatie](/course/open/nn-initialization?lang=nl) erin.",
            "In `forward`: eerst `z = self.hidden[k](h)`; de batch normalization alleen onder `if self.norms is not None:`; dan de ReLU; en een `if` en `else` op `self.residual` voor wat `h` wordt.",
        ],
        why="`list(self.hidden)` maakt van de `nn.ModuleList` een lijst die `+` met de andere twee layers kan samenvoegen. In `forward` is `h` wat doorgaat naar de volgende layer: met een residual connection houdt hij zichzelf en komt `z` erbij, zodat het signaal en de gradient elke layer heel kunnen passeren.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het vervagende signaal uit [Vanishing en exploding gradients](/course/open/nn-vanishing-exploding?lang=nl) hield de 10 layers tegen. He initialization uit [Initialisatie](/course/open/nn-initialization?lang=nl), [Batch normalization](/course/open/nn-batchnorm?lang=nl) en [Residual connections](/course/open/nn-residual?lang=nl) lossen het elk op, en je Trainer uit [Een herbruikbare training loop](/course/open/nn-trainer?lang=nl), met AdamW, doet de rest. Je project heeft nu `overfit_one_batch` en `DeepMLP` naast de Trainer. Met *Toon verdieping* aan komt eerst *Een draaiboek voor tunen*: daar vind je een goede learning rate met een range test, en verander je daarna één hyperparameter tegelijk, in een kleine sweep. In de volgende les van de hoofdroute, *Een neural language model dat verder terugkijkt*, krijgt je language model een MLP dat de laatste paar tokens tegelijk leest.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_overfit_first, result_overfit, result_deep):
    L.wrap_up(choice_overfit_first, result_overfit, result_deep)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
