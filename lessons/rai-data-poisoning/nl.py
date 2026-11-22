# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-data-poisoning. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-data-poisoning", "nl")
    C = L.checks   # de cijfers van QMNIST, het netwerk van torch-mlp-digits, en labels die de course omdraait


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over waar deze les op voortbouwt.
    """)
    return


@app.cell(hide_code=True)
def quick_check_stage():
    quick_stage = L.choice(
        "quick-stage",
        question="NIST deelt aanvallen op machine learning in naar de fase waarin ze toeslaan. Wanneer slaat een poisoning-aanval toe?",
        options={
            "use": "Als het model gebruikt wordt, via een input die het leest",
            "right": "Terwijl het model leert, via de training data",
            "file": "Na de training, door het bestand met weights te veranderen",
        },
        correct="right",
        why={
            "right": "NIST onderscheidt de twee families naar de fase: poisoning slaat toe terwijl het model leert, evasion als het gebruikt wordt. Zie *Wanneer: terwijl het leert, of als het gebruikt wordt* in [Aanvallen op machine learning](/course/open/rai-ml-threat-models?lang=nl).",
            "use": "Dat is een evasion attack, zoals de stickers op het stopbord: het model blijft zoals het getraind is. Zie *Wanneer: terwijl het leert, of als het gebruikt wordt* in [Aanvallen op machine learning](/course/open/rai-ml-threat-models?lang=nl).",
            "file": "Een poisoning-aanval heeft de weights niet nodig: die werkt via de data waarvan het model leert. Zie *Wanneer: terwijl het leert, of als het gebruikt wordt* in [Aanvallen op machine learning](/course/open/rai-ml-threat-models?lang=nl).",
        },
    )
    quick_stage
    return


@app.cell(hide_code=True)
def quick_check_argmax():
    quick_argmax = L.choice(
        "quick-argmax",
        question="DigitsMLP geeft 10 logits voor elke afbeelding. Welk cijfer leest het?",
        options={
            "sum": "De som van de tien logits, afgerond op een heel cijfer",
            "half": "Het eerste cijfer waarvan de logit boven 0,5 ligt",
            "right": "Het cijfer waarvan de logit het grootst is",
        },
        correct="right",
        why={
            "right": "`model(X_test).argmax(dim=1)` neemt voor elke afbeelding de plaats van de grootste logit, en die plaats is het cijfer. Zie het uitgewerkte voorbeeld van [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl).",
            "sum": "De tien logits zijn scores, één voor elk cijfer; hun som zegt niets over welk cijfer het is. Het cijfer is de plaats van de grootste: `argmax`. Zie [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl).",
            "half": "Logits zijn geen kansen, en er kunnen er meerdere boven 0,5 liggen. Het cijfer is de plaats van de grootste: `argmax`. Zie [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl).",
        },
    )
    quick_argmax
    return


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een model leert wat de labels zeggen. De verzonnen bibliotheek wil een netwerk dat de handgeschreven nummers op de retourbonnen leest, en vraagt vrijwilligers om daarvoor 6000 handgeschreven cijfers te labelen. Een van hen wil misschien dwarsliggen. Die vrijwilliger kan niet bij het netwerk of de code, en kan alleen verkeerde labels schrijven. Hoeveel verkeerde labels zijn er nodig, en maakt het uit welke? In deze les draai je zelf labels van de cijfers om, train je het netwerk van [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl) erop, en meet je wat het leest, cijfer voor cijfer. Niets verlaat je computer: je verandert je eigen kopie van de labels.
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="Getraind op de schone labels leest het netwerk 929 van de 1000 testcijfers goed. Nu wordt 30% van de labels willekeurig veranderd, elk in een ander cijfer. Hoeveel leest het netwerk na dezelfde training goed?",
        options={
            "fewer": "Ongeveer 650: 30% minder",
            "right": "Nog meer dan 850",
            "chance": "Ongeveer 100, zoveel als gokken",
        },
        reveal="**881.** 1792 van de 6000 labels willekeurig omdraaien kost 48 testcijfers. Het uitgewerkte voorbeeld laat het zien, en *Het idee* zegt waarom een netwerk zoveel ruis aankan.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Wat poisoning is

    Het rapport van NIST over adversarial machine learning omschrijft het ruim: poisoning-aanvallen zijn aanvallen tijdens de trainingsfase van een algoritme voor machine learning. Bij data poisoning heeft een aanvaller een deel van de training data in handen, en voegt trainingsvoorbeelden toe of verandert ze. Het is niet nieuw: de eerste bekende poisoning-aanval, uit 2006, was gericht op het maken van handtekeningen waarmee computerwormen herkend worden.

    ### Overal slechter, of juist daar fout

    Poisoning kan twee doelen van [Aanvallen op machine learning](/course/open/rai-ml-threat-models?lang=nl) dienen. Een aanval op de beschikbaarheid maakt een model volgens NIST meestal slechter op alle voorbeelden, zonder onderscheid; *targeted* en *backdoor poisoning* schenden de integriteit bij een klein aantal voorbeelden die de aanvaller kiest. De eerste soort zie je makkelijk: volgens NIST valt die meestal op als je de gewone scores van een model in de gaten houdt. De tweede zie je alleen bij de inputs die de aanvaller koos.

    ### Label flipping

    De eenvoudigste weg naar binnen loopt via de labels. NIST noemt label flipping een eenvoudige black-box-aanval: de aanvaller maakt trainingsvoorbeelden met verkeerde of veranderde labels. Om een model als geheel te bederven kan daar een groot deel van de data voor nodig zijn. Tegen een doelwit is het makkelijk: wie bepaalt welke labels de training data krijgt, heeft volgens NIST aan label flipping een doeltreffende gerichte aanval. De aanvaller voegt een paar vergiftigde voorbeelden met het gewenste label toe, en het model leert het verkeerde label.

    ### Willekeurig omgedraaid, gekozen omgedraaid

    Welke labels je omdraait, telt zwaarder dan hoeveel. Xiao, Xiao en Eckert (2012) draaiden labels om voor een support vector machine, een ander soort classifier, die [De breedste straat: de maximum-margin classifier](/course/open/ml-max-margin?lang=nl) uitlegt. Bij 20 willekeurig omgedraaide labels veranderde de foutkans nauwelijks. Met 20 labels die hun methode koos, 10% van de training data, steeg de foutkans naar 50%: niet beter dan gokken. Biggio, Nelson en Laskov (2012) gingen verder. Voor een support vector machine die twee MNIST-cijfers uit elkaar houdt, zoals 7 en 1, met 100 trainingsafbeeldingen, berekenden ze één trainingspunt: dat ene punt liet de fout stijgen van 2 tot 5% naar 15 tot 20%.

    Neural networks kunnen veel willekeurige ruis aan. Rolnick en collega's (2017) voegden extra voorbeelden met willekeurige labels toe aan MNIST. Bij een convolutional network, een soort netwerk voor afbeeldingen, melden ze dat het zelfs met 100 labels met ruis voor elk goed label nog 91% haalde. Ze verklaren het met een marge: diepe neural networks zijn robuust, zelfs tegen ruis met een structuur, zolang het goede label met minstens een kleine marge het waarschijnlijkst blijft. Hun ruis kwam bovenop elk schoon voorbeeld. Een deel omdraaien haalt schone voorbeelden weg, en ze waarschuwen dat een kleiner aantal goede voorbeelden het leren meer schaadt dan extra labels met ruis.

    De marge is rekenwerk. Draai een deel $p$ van de labels willekeurig om, elk naar een van de negen andere cijfers: van de afbeeldingen van elk cijfer houdt $1 - p$ het eigen label, en krijgt $p / 9$ elk ander label. Het goede label blijft het vaakst zolang $1 - p > p / 9$, dus zolang $p < 0{,}9$. Het plaatje hieronder tekent beide lijnen.

    ### Een backdoor is iets anders

    Een backdoor werkt anders: de aanvaller voegt aan een paar trainingsinputs een trigger toe en geeft ze het label dat de aanvaller wil. Gu, Dolan-Gavitt en Garg (2017) trainden BadNets met triggers zoals één felle pixel in de rechteronderhoek van de afbeelding. Op MNIST las een BadNet meer dan 99% van de afbeeldingen met een trigger verkeerd, en de aanval werkte al als zulke afbeeldingen maar 10% van de training data waren. Zonder de trigger leest het bijna even goed als eerst, dus schone testcijfers laten het niet zien. Deze les draait alleen labels om.

    ### Op de schaal van het web

    Grote modellen leren van meer data dan iemand met de hand nakijkt. Carlini en collega's (2024) lieten zien dat ze in 2023, door oude webdomeinen te kopen voor 60 dollar, 0,01% van LAION-400M of COYO-700M hadden kunnen vergiftigen, twee grote verzamelingen afbeeldingen van het web. Ze deden het niet: ze vergiftigden geen enkele dataset echt. Souly en collega's (2025) vergiftigden language models tijdens hun training met documenten die de modellen leren om na een trigger wartaal te schrijven: 250 vergiftigde documenten tastten modellen van elke grootte die ze probeerden even sterk aan, hoeveel data ze ook kregen. Bij hun model van 13 miljard parameters was dat 0,00016% van de training tokens, bij een van 600 miljoen 0,0035%. De wartaal is onschuldig, en de blogpost bij de paper voegt toe dat nog onduidelijk is hoe ver dit doorgaat bij nog grotere modellen.

    ### Hoe zeker is een score?

    Een score op testcijfers is ook een meting. Yadav en Bottou (2019), die MNIST opnieuw opbouwden als QMNIST, de cijfers van deze course, schrijven dat een foutkans van ongeveer 1,0% op de gebruikelijke 10.000 testvoorbeelden gerapporteerd wordt als 1% ± 0,2%: 100 ± 20 fouten. Die foutmarge is twee standaardfouten. De standaardfout van een aandeel $s$, gemeten op $n$ testafbeeldingen, is $\sqrt{s(1 - s)/n}$: hoeveel het aandeel zou schommelen als je opnieuw meet op andere testafbeeldingen. [Hoe zeker is een getal? Betrouwbaarheidsintervallen en de bootstrap](/course/open/stats-bootstrap?lang=nl) gaat er dieper op in. De foutmarge is het dubbele, $2\sqrt{s(1 - s)/n}$. Op de 1000 testcijfers van de course heeft een score van 0,929 een foutmarge van 0,016; op de 99 testafbeeldingen van de 7'en is de foutmarge van hun score 0,048. Een daling kleiner dan de foutmarge kan toeval zijn.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | | |
        |---|---|
        | **data poisoning** | een aanval via de training data: de aanvaller voegt trainingsvoorbeelden toe of verandert ze |
        | **label flipping** | data poisoning die alleen labels verandert: willekeurig, of van één cijfer naar een ander |
        | **targeted poisoning** | poisoning waardoor het model zich vergist bij een paar inputs die de aanvaller koos, terwijl de rest blijft zoals het was |
        | **backdoor** | verborgen gedrag dat een trigger in de input aanzet |
        | **foutmarge** | twee standaardfouten, zoals de paper van QMNIST die gebruikt: hoeveel een score op $n$ testafbeeldingen door toeval kan afwijken |
        | $s$, $n$ | een aandeel gemeten op testafbeeldingen, en het aantal testafbeeldingen |
        """),
        L.watch_out("Een totaalscore zegt niet waar een model faalt. Het netwerk van oefening 1, dat de 7'en als 1 leerde, leest nog 840 van de 1000 testcijfers goed, 89 minder dan schoon: het verliest 90 van de 7'en, en leest de andere cijfers even goed als het schone netwerk."),
        mo.Html(C.idea_picture("nl")),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Bewaar de schone labels: draai een kopie om, zodat je op allebei kunt trainen.
    2. Draai om: een deel van alle labels willekeurig, of het eerste deel van één cijfer naar een ander.
    3. Train hetzelfde netwerk met dezelfde seeds op de omgedraaide labels.
    4. Meet het op de schone testcijfers: in totaal, en per cijfer.
    5. Vergelijk elke score met de schone: een daling groter dan de foutmarge is meer dan toeval.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 1 tot en met 4 voor willekeurig omdraaien. `flip_random(labels, share, seed)` geeft een kopie waarin elk label met kans `share` 1 tot 9 plaatsen opschuift, na 9 weer terug naar 0, zodat het een ander cijfer wordt. `% 10` is de rest na delen door 10, dus 9 + 3 wordt 2. `.astype(np.int64)` maakt van de labels hele getallen van 64 bits, het type dat de loss wil, en `.numpy()` maakt van wat het netwerk leest weer een NumPy-array. Het netwerk en de training zijn die van [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl): `DigitsMLP(64)`, 10 epochs van mini-batches van 64, met $\eta$ = 0,5, elke keer met dezelfde seeds. De cel traint het twee keer, op de schone labels en met 30% ervan omgedraaid, en telt de testcijfers die elk netwerk goed leest. Dat duurt ongeveer twee seconden.
    """)
    return


@app.cell
def example_flip():
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    # 1. de cijfers, en de schone labels
    data = np.load("data/qmnist-small.npz")
    X = torch.tensor(data["train_images"], dtype=torch.float32) / 255
    labels = data["train_labels"].astype(np.int64)
    X_test = torch.tensor(data["test_images"], dtype=torch.float32) / 255
    test_labels = data["test_labels"].astype(np.int64)


    # 2. een kopie waarin elk label met kans share een van de negen andere cijfers wordt
    def flip_random(labels, share, seed):
        rng = np.random.default_rng(seed)
        flipped = labels.copy()
        picked = rng.random(len(labels)) < share
        shifts = rng.integers(1, 10, size=len(labels))   # 1 tot 9 plaatsen verder: nooit hetzelfde cijfer
        flipped[picked] = (labels[picked] + shifts[picked]) % 10
        return flipped


    class DigitsMLP(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.hidden = nn.Linear(784, hidden)
            self.out = nn.Linear(hidden, 10)

        def forward(self, x):
            return self.out(torch.relu(self.hidden(x)))


    # 3. hetzelfde netwerk en dezelfde seeds, op labels y; het geeft wat het leest voor elk testcijfer
    def train(y):
        torch.manual_seed(0)
        model = DigitsMLP(64)
        opt = torch.optim.SGD(model.parameters(), lr=0.5)
        loss_fn = nn.CrossEntropyLoss()
        torch.manual_seed(2)
        loader = DataLoader(TensorDataset(X, torch.tensor(y)), batch_size=64, shuffle=True)
        for epoch in range(10):
            for xb, yb in loader:
                loss = loss_fn(model(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            return model(X_test).argmax(dim=1).numpy()


    # 4. de schone labels, en 30% ervan willekeurig omgedraaid
    flipped = flip_random(labels, 0.3, 1)
    print("labels changed:", np.sum(flipped != labels), "of", len(labels))
    for name, y in [("clean", labels), ("30% flipped", flipped)]:
        pred = train(y)
        print(name, "right:", np.sum(pred == test_labels), "of", len(test_labels))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Met 30% willekeurig veranderen 1792 van de 6000 labels (*labels changed*). Het schone netwerk leest 929 van de 1000 testcijfers goed, en het netwerk dat op de omgedraaide labels trainde 881: 48 minder, terwijl bijna één label op de drie fout was. De course trainde hetzelfde netwerk ook op andere aandelen, met dezelfde `flip_random` en seed 1:

    | deel willekeurig omgedraaid | 0 | 0,1 | 0,2 | 0,3 | 0,5 | 0,7 | 0,9 |
    |---|---|---|---|---|---|---|---|
    | labels veranderd | 0 | 602 | 1203 | 1792 | 2982 | 4181 | 5386 |
    | testcijfers goed | 929 | 900 | 887 | 881 | 788 | 614 | 102 |

    Tot de helft van de labels blijft het netwerk het meeste goed lezen. Bij 0,9, waar het goede label niet vaker voorkomt dan elk fout label, leest het er 102 goed, ongeveer zoveel als gokken. Oefening 1 draait in plaats daarvan de labels van één cijfer om, en oefening 2 meet wat dat met elk cijfer doet.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_flip():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. flip_digit: alleen de 7'en

        Een aanvaller die wil dat het netwerk één cijfer verkeerd leest, draait alleen dat cijfer om. `flip_digit(labels, source, target, share)` geeft een kopie van `labels` waarin het eerste deel `share` van de afbeeldingen met label `source` het label `target` krijgt, een aantal dat je afrondt met `round`; elk ander label blijft zoals het was. `flipped = labels.copy()` maakt de kopie. `np.arange(len(labels))` nummert de plaatsen 0, 1, 2 enzovoort; een mask houdt de plaatsen over waar het label `source` is. Vervang de `...`. De check roept `flip_digit(LABELS, 7, 1, 0.8)` aan, en daarna andere cijfers en aandelen.
        """),
        L.task("Geef het eerste deel share van de afbeeldingen met label source het label target, in een kopie van labels."),
    ])
    return


@app.cell(hide_code=True)
def exercise_flip():
    box_flip = L.code_box("flip-digit", lines=10)
    box_flip
    return (box_flip,)


@app.cell(hide_code=True)
def check_flip(box_flip):
    result_flip = L.check("flip-digit", box_flip.value)
    result_flip
    return (result_flip,)


@app.cell(hide_code=True)
def hints_flip():
    L.hints(
        "flip-digit",
        [
            "Stap 2 van het recept, voor één cijfer: zoek de plaatsen waar het label source is, neem het eerste deel daarvan, en geef die het label target.",
            "Met een mask op de genummerde plaatsen krijg je de plaatsen van source. Hoeveel je omdraait, is `share` keer het aantal plaatsen, afgerond met `round`; een slice vanaf het begin neemt er zoveel.",
            "`places = np.arange(len(labels))[labels == source]`, `count = round(share * len(places))`, dan `flipped[___] = target`",
        ],
        why="`np.arange(len(labels))[labels == source]` houdt de plaatsen van de afbeeldingen met label `source` over, op volgorde. `round(share * len(places))` is hoeveel daarvan je omdraait, en `flipped[places[:count]] = target` geeft de eerste `count` ervan het label `target` in de kopie; `labels` blijft zoals het was. Met `flip_digit(LABELS, 7, 1, 0.8)` krijgen 521 van de 651 zevens het label 1.",
    )
    return


@app.cell(hide_code=True)
def text_after_flip():
    mo.md(r"""
    Dat zijn 521 labels, minder dan de 602 die 10% willekeurig verandert. De course trainde het netwerk erop: het leest 840 van de 1000 testcijfers goed. Van de 99 test-7'en leest het er 3 goed en 89 als een 1. De 901 andere testcijfers leest het even goed als het schone netwerk: 837 ervan, tegen 836. De 521 labels kosten dus 89 testcijfers, meer dan de 1792 willekeurig omgedraaide labels (48), omdat de 7'en een tiende van de testcijfers zijn. De totaalscore laat zien dát er iets mis is, maar niet wat.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_per():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. per_digit: een score per cijfer

        Eén score verbergt waar een netwerk faalt. `per_digit(pred, labels)` krijgt wat een netwerk voor elke testafbeelding leest, `pred`, en de goede cijfers, `labels`, en geeft een lijst van tien aandelen: voor elk cijfer van 0 tot en met 9 het deel van de afbeeldingen met dat label dat `pred` goed leest. Vervang de `...`. De check roept `per_digit(GUESSES, TRUTH)` aan, twintig verzonnen gokken voor twintig afbeeldingen, twee van elk cijfer, en daarna andere gokken.
        """),
        L.task("Geef voor elk cijfer het deel van de afbeeldingen ervan dat pred goed leest."),
    ])
    return


@app.cell(hide_code=True)
def exercise_per():
    box_per = L.code_box("per-digit", lines=10)
    box_per
    return (box_per,)


@app.cell(hide_code=True)
def check_per(box_per):
    result_per = L.check("per-digit", box_per.value)
    result_per
    return (result_per,)


@app.cell(hide_code=True)
def hints_per():
    L.hints(
        "per-digit",
        [
            "Stap 4 van het recept, per cijfer: kijk voor elk cijfer alleen naar de afbeeldingen waarvan het label dat cijfer is.",
            "Een mask met `labels == digit` houdt die afbeeldingen over. Van hun gokken in `pred` is het deel dat ook dat cijfer is het gemiddelde van `True` en `False`, en dat geeft `np.mean`.",
            "`mine = labels == digit` in de lus over `range(10)`, dan `shares.append(float(np.mean(___ == digit)))`",
        ],
        why="`for digit in range(10):` gaat de cijfers langs; `mine = labels == digit` is `True` bij de afbeeldingen met dat label, `pred[mine]` zijn hun gokken, en `np.mean(pred[mine] == digit)` is het deel daarvan dat klopt. `float` maakt er een gewoon getal van. `per_digit(GUESSES, TRUTH)` geeft 0,5 voor de 7'en en de 9'en en 1,0 voor elk ander cijfer.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_verdict():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. verdict: een daling, of toeval?

        `verdict(clean, poisoned, n)` krijgt twee scores op dezelfde `n` testafbeeldingen: die van het schone netwerk en die van een vergiftigd netwerk. `verdict` geeft `"worse"` als `poisoned` meer dan de foutmarge van `clean` lager is dan `clean`, en anders `"chance"`. De foutmarge is $2\sqrt{\text{clean}\,(1 - \text{clean})/n}$, met `math.sqrt`. Het is een grove regel: die houdt er geen rekening mee dat de vergiftigde score ook een meting is. Vervang de twee `...`. De check roept `verdict(0.929, 0.881, 1000)` aan, het schone netwerk tegen dat met 30% willekeurig omgedraaid, en daarna andere scores.
        """),
        L.task("Geef \"worse\" als poisoned meer dan de foutmarge onder clean ligt, en anders \"chance\"."),
    ])
    return


@app.cell(hide_code=True)
def exercise_verdict():
    box_verdict = L.code_box("verdict", lines=9)
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
            "Stap 5 van het recept: reken de foutmarge van de schone score uit, en kijk dan of de daling groter is.",
            "De foutmarge is 2 keer de wortel van clean keer 1 min clean, gedeeld door n. Alleen een daling telt, geen stijging.",
            "`margin = 2 * math.sqrt(clean * (1 - clean) / n)`, dan `if ___ > margin:` met `return \"worse\"` eronder",
        ],
        why="`2 * math.sqrt(clean * (1 - clean) / n)` is twee standaardfouten van de schone score. `clean - poisoned` is de daling; als die groter is dan de foutmarge, geeft `verdict` `\"worse\"`, en anders geeft de laatste regel `\"chance\"`. Voor 0,929 op 1000 testafbeeldingen is de foutmarge 0,016, dus 0,881 is worse.",
    )
    return


@app.cell(hide_code=True)
def text_after_verdict():
    mo.md(r"""
    Cijfer voor cijfer laat het plaatje de twee aanvallen uit elkaar zien. Met 80% van de 7'en als 1 zijn alleen de 7'en slechter. Met 30% willekeurig zijn het er zes, en geen ervan daalt bij lange na zoveel als de 7'en: het slechtst gaat het bij de 3'en: 0,757 goed, tegen 0,925 bij het schone netwerk.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Vergiftig het zelf

    Kies hoe je de labels omdraait, willekeurig of alleen de 7'en als 1, en welk deel. De 7'en gaan door jouw `flip_digit` zodra oefening 1 klopt, en tot dan door die van de course. De knop traint het netwerk op die labels zoals het uitgewerkte voorbeeld doet, in een paar seconden. Tijdens de training laat de grafiek elke 10 mini-batches zien welk deel van de testcijfers het fout leest, met het schone netwerk gestreept; ernaast zie je welk deel van elk cijfer het goed leest, en wat het leest voor één testcijfer van elke soort. Daarna kun je terug door de epochs.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_kind = mo.ui.dropdown(options={"willekeurig": "random", "alleen de 7'en, als 1": "sevens"}, value="alleen de 7'en, als 1",
                               label="Omdraaien")
    demo_share = L.slider("demo-share", "deel omgedraaid", start=0.8, low=0.0, high=1.0, step=0.1, role="hyper")
    demo_button = mo.ui.run_button(label="Train op deze labels")
    mo.vstack([mo.hstack([demo_kind, demo_share], justify="start", gap=1.5), demo_button])
    return demo_button, demo_kind, demo_share


@app.cell(hide_code=True)
def demo_run(box_flip, demo_button, demo_kind, demo_share, result_flip):
    if not demo_button.value:
        _out = mo.md("*Kies de labels, en druk op de knop om te trainen.*")
    else:
        _kind, _share = demo_kind.value, round(float(demo_share.widget.value), 1)
        _yours = C.mine(box_flip.value, result_flip.state == "correct", "flip_digit") if _kind == "sevens" else None
        _what = f"{round(100 * _share)}% van de labels willekeurig omgedraaid" if _kind == "random" else f"{round(100 * _share)}% van de 7'en als 1 gelabeld"
        _words = {"title": f"Getraind met {_what}", "x": "epoch", "y": "deel testcijfers fout", "digit": "cijfer",
                  "clean": "schone labels", "now": "nu", "guess": "Wat het netwerk nu leest",
                  "run": "deze labels", "after": "na epoch {e}", "guess_after": "Wat het las na epoch {e}",
                  "alt": f"Het deel van de 1000 testcijfers dat het netwerk fout leest, elke 10 mini-batches, getraind met {_what}; gestreept het netwerk dat op de schone labels trainde."}
        _now = {}

        def _progress(_p):
            if "live" not in _now:
                _now["live"] = L.live({"title": _words["title"], "x": "epoch", "y": _words["y"], "alt": _words["alt"]},
                                      C.EPOCHS, low=0.0, high=1.0)
                _now["live"].reference(C.course_run("random", 0.0)[0], "schone labels", role="loss")
            _now["live"].watch(lambda: C.now_parts(_p["pred"], _words, L.lang))
            _now["live"].add(_p["x"], _p["wrong"], f"**{_words['title']}** · epoch {_p['epoch']} van {C.EPOCHS} · {round(1000 * (1 - _p['wrong']))} van 1000 testcijfers goed")

        mo.output.replace(mo.md("*Epoch 1 · begint …*"))
        _run = C.demo(_kind, _share, _yours, _progress)
        if isinstance(_run, str):
            _note = {"flip": "*Je `flip_digit` stopt met een fout op de labels van de training, of geeft geen 6000 labels van 0 tot en met 9. Het plaatje onder oefening 1 zegt wat er misgaat.*",
                     "slow-flip": "*Je `flip_digit` doet er langer dan tien seconden over op de 6000 labels. Gaat die maar één keer over de labels?*",
                     "slow": "*De training duurt langer dan twee minuten. Is de computer met iets anders bezig?*"}[_run]
            _out = _now["live"].stopped(_note) if "live" in _now else mo.md(_note)
        else:
            _labels, _curve, _epochs = _run
            _pred = _epochs[-1]
            _right = round(1000 * C.accuracy(_pred))
            _changed = int((_labels != C.LABELS.value).sum())
            _base, _shares, _n = C.per_digit(C.clean(), C.TEST_LABELS.value), C.per_digit(_pred, C.TEST_LABELS.value), C.counts()
            _worse = [str(_d) for _d in range(10) if C.verdict(_base[_d], _shares[_d], _n[_d]) == "worse"]
            _which = "de " + ", ".join(_d + "'en" for _d in _worse[:-1]) + (" en " if len(_worse) > 1 else "") + (_worse[-1] + "'en" if _worse else "")
            _drop = (f"Volgens `verdict` van de course daalt de score meer dan de foutmarge bij {_which}." if _worse
                     else "Volgens `verdict` van de course daalt bij geen enkel cijfer de score meer dan de foutmarge.")
            _who = " Je `flip_digit` maakte de labels." if _yours else ""
            _out = mo.vstack([
                mo.md(f"""Getraind met {_what}: {_changed:d} van de 6000 labels veranderd. Het netwerk leest <span class="ea-num">{_right}</span> van de 1000 testcijfers goed, tegen {round(1000 * C.accuracy(C.clean()))} bij de schone labels. {_drop}{_who}"""),
                L.explore("rewind", "epoch", lambda _e: (C.rewind_frame(_curve, _epochs, int(_e), _words, L.lang),
                                                         f"Na epoch {int(_e)}: {round(1000 * C.accuracy(_epochs[int(_e) - 1]))} van de 1000 testcijfers goed."),
                          start=C.EPOCHS, values=list(range(1, C.EPOCHS + 1)), role="",
                          task=("Ga terug naar epoch 1, en dan weer vooruit: wanneer begint het netwerk de 7'en als 1 te lezen?" if _kind == "sevens"
                                else "Ga terug naar epoch 1, en dan weer vooruit: welke cijfers leest het als eerste goed?")),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_noise():
    mo.md(r"""
    ### Oefening 4. Waarom zo weinig?

    Rolnick en collega's hielden op MNIST 91% over met 100 willekeurige labels voor elk goed label. Het netwerk van deze les leest, met 90% van de labels willekeurig omgedraaid, 102 van de 1000 testcijfers goed.
    """)
    return


@app.cell(hide_code=True)
def choice_noise_cell():
    choice_noise = L.choice(
        "noise",
        question="Wat verklaart het verschil?",
        options={
            "right": "Bij 0,9 komt het goede label niet vaker voor dan elk fout label",
            "big": "Hun netwerk was groter, en daardoor kon het de foute labels gewoon negeren",
            "easy": "Hun MNIST-cijfers waren makkelijker te lezen dan QMNIST",
        },
        correct="right",
        why={
            "right": "Met 100 willekeurige labels voor elk goed label blijft elk schoon voorbeeld, en komt het goede label nog net vaker voor: 11 van de 101 labels zijn goed, tegen 10,1 bij puur toeval. 90% willekeurig omdraaien haalt negen op de tien schone voorbeelden weg, en dan komt het goede label even vaak voor als elk van de negen andere: de marge is weg.",
            "big": "De marge geeft de doorslag: netwerken leren het goede label zolang dat met minstens een kleine marge het waarschijnlijkst blijft. Bij 0,9 is dat niet meer zo. Zie *Willekeurig omgedraaid, gekozen omgedraaid* in *Het idee*.",
            "easy": "QMNIST bouwt MNIST opnieuw op: de training set bevat 60.000 afbeeldingen die elk overeenkomen met een trainingsafbeelding van MNIST. Het verschil zit in de labels. Zie *Willekeurig omgedraaid, gekozen omgedraaid* in *Het idee*.",
        },
    )
    choice_noise
    return (choice_noise,)


@app.cell(hide_code=True)
def text_exercise_count():
    mo.md(r"""
    ### Oefening 5. Een aandeel, of een aantal?

    Souly en collega's vergiftigden language models van 600 miljoen en van 13 miljard parameters, elk met 250 documenten.
    """)
    return


@app.cell(hide_code=True)
def choice_count_cell():
    choice_count = L.choice(
        "count",
        question="Wat betekent dat voor een aanvaller op het web?",
        options={
            "share": "Het aandeel in de data telt, dus een groot model heeft er veel meer nodig",
            "right": "Een vast aantal documenten kan genoeg zijn, hoe groot de data ook is",
            "lost": "Grote modellen leren van zoveel tekst dat 250 documenten erin verdwijnen",
        },
        correct="right",
        why={
            "right": "250 documenten waren 0,00016% van de training tokens van het model van 13 miljard parameters en 0,0035% van dat van 600 miljoen, en allebei leerden ze de backdoor. Het was onschuldige wartaal, en of dit ook geldt voor nog grotere modellen, is nog onduidelijk, schrijft de blogpost bij de paper.",
            "share": "Ouder werk mat het vergif als aandeel van de training data; Souly en collega's vonden dat het aantal telt: dezelfde 250 documenten werkten bij elke grootte die ze probeerden. Zie *Op de schaal van het web* in *Het idee*.",
            "lost": "Dat zou je verwachten, en het is niet wat Souly en collega's vonden: 250 documenten tastten modellen van elke grootte aan, ook al trainden de grootste op meer dan 20 keer zoveel schone data. Zie *Op de schaal van het web* in *Het idee*.",
        },
    )
    choice_count
    return (choice_count,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Aanvallen op machine learning](/course/open/rai-ml-threat-models?lang=nl) veranderde een aanvaller een input na de training; hier veranderde de aanvaller de training data ervoor. Allebei volgen ze de doelen van NIST: willekeurig omdraaien richt zich op de beschikbaarheid, één cijfer omdraaien op de integriteit. Het netwerk is dat van [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl), met dezelfde seeds, dus elk verschil dat je mat, komt van de labels. De volgende lessen van deze module vallen language models aan, te beginnen met *Hoe vaak werkt een injection?*

    **Een voorbeeld uit het dagelijks leven.** Een webwinkel met reviews. Honderd nepreviews van één ster, verspreid over alle producten, veranderen het gemiddelde van geen enkel product veel; dezelfde honderd op één product laten dat product zakken. Het gemiddelde over alle reviews van de winkel daalt in beide gevallen evenveel: alleen het gemiddelde per product laat zien welk product geraakt is.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_count, choice_noise, result_flip, result_per, result_verdict):
    L.wrap_up(result_flip, result_per, result_verdict, choice_noise, choice_count)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
