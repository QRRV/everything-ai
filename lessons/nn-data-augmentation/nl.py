# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nn-data-augmentation. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow verschuiving, verschuivingen

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-data-augmentation", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een afbeelding is een rooster van getallen](/course/open/data-images-as-arrays?lang=nl) en [Loss curves lezen](/course/open/torch-loss-curves?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_pixel():
    quick_pixel = L.choice(
        "quick-pixel",
        question="`img` is een grijswaardenbeeld van 28 bij 28 pixels, als array met shape (28, 28). Wat is `img[3, 10]`?",
        options={
            "column": "De pixel in kolom 3, rij 10",
            "right": "De pixel in rij 3, kolom 10",
            "count": "Het aantal pixels in rij 3 tot kolom 10",
        },
        correct="right",
        why={
            "column": "De eerste index telt de rijen, van boven af. Zie [Een afbeelding is een rooster van getallen](/course/open/data-images-as-arrays?lang=nl).",
            "count": "Een index geeft één getal, de helderheid van één pixel. Zie [Een afbeelding is een rooster van getallen](/course/open/data-images-as-arrays?lang=nl).",
            "right": "Axis 0 loopt langs de rijen omlaag, axis 1 langs de kolommen opzij: `img[3, 10]` is de helderheid van de pixel in rij 3, kolom 10, allebei geteld vanaf 0.",
        },
    )
    quick_pixel
    return (quick_pixel,)


@app.cell(hide_code=True)
def quick_check_gap():
    quick_gap = L.choice(
        "quick-gap",
        question="De training loss van een model blijft dalen, maar zijn validation loss stijgt na epoch 12 weer. Waarom stijgt de validation loss?",
        options={
            "right": "Het leert zijn trainingsvoorbeelden uit zijn hoofd",
            "large": "De learning rate is te groot: de stappen schieten door",
            "harder": "De cijfers van de validation set zijn moeilijker dan de trainingscijfers",
        },
        correct="right",
        why={
            "large": "Een te grote learning rate laat ook de training loss springen of groeien, en die blijft hier dalen. Zie [Loss curves lezen](/course/open/torch-loss-curves?lang=nl).",
            "harder": "Moeilijkere cijfers geven vanaf het begin een hogere validation loss, niet een die eerst daalt en dan weer stijgt. Zie *Overfitting* in [Loss curves lezen](/course/open/torch-loss-curves?lang=nl).",
            "right": "Overfitting: na epoch 12 wordt het model beter in zijn eigen voorbeelden en slechter in nieuwe. Deze les gaat over een manier om dat moeilijker te maken.",
        },
    )
    quick_gap
    return (quick_gap,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Loss curves lezen](/course/open/torch-loss-curves?lang=nl) begon een MLP na ongeveer 8 epochs zijn 1.000 trainingscijfers uit zijn hoofd te leren. Meer trainingsvoorbeelden maken dat moeilijker, maar nieuwe voorbeelden kosten tijd en geld: iemand moet ze schrijven en labelen. Toch is een cijfer dat één pixel verder naar rechts geschreven is, nog steeds hetzelfde cijfer. Kun je van de voorbeelden die je hebt gratis nieuwe maken?

    **Een voorbeeld uit het dagelijks leven.** Je leert een nieuw handschrift lezen van maar één bladzijde. Na een tijdje ken je die bladzijde uit je hoofd, woord voor woord, en toch struikel je over de volgende, waar dezelfde letters iets hoger of lager op de regel staan, of iets meer naar links. Meer bladzijden in dat handschrift zouden helpen, maar die zijn er niet.
    """)
    return


@app.cell(hide_code=True)
def predict_shift():
    _epoch0, _loss0 = C.best(0)
    _epoch1, _loss1 = C.best(1)
    guess_shift = L.guess(
        "shift-loss",
        question=f"De run uit [Loss curves lezen](/course/open/torch-loss-curves?lang=nl) bereikt zijn laagste validation loss, {L.number(_loss0, 2)}, na epoch {_epoch0}. Nu wordt elk trainingscijfer, elke keer dat het gebruikt wordt, eerst willekeurig verschoven, hooguit 1 pixel omhoog of omlaag en hooguit 1 pixel naar links of rechts. De rest blijft hetzelfde. Wat denk je dat de laagste validation loss doet?",
        options={
            "same": "Die blijft ongeveer gelijk: de cijfers zijn dezelfde",
            "right": "Die wordt veel lager",
            "higher": "Die wordt hoger: het model leert van rommeligere voorbeelden",
        },
        reveal=f"**Die wordt veel lager.** Met de verschuivingen is de validation loss het laagst na epoch {_epoch1}, op {L.number(_loss1, 2)}, tegen {L.number(_loss0, 2)} zonder. Het model ziet elk cijfer in tot negen standen, elke epoch een nieuw getrokken stand, dus kan het de precieze pixels van zijn 1.000 voorbeelden nauwelijks uit het hoofd leren. Het moet leren hoe een 3 eruitziet, waar die ook geschreven is, en dat werkt ook op nieuwe cijfers.",
    )
    guess_shift
    return (guess_shift,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Data augmentation** maakt nieuwe trainingsvoorbeelden van de voorbeelden die je hebt, met kleine veranderingen waarbij het label blijft kloppen: een cijfer dat een pixel is verschoven, is nog steeds hetzelfde cijfer. De veranderingen zijn willekeurig, en nieuw elke keer dat een voorbeeld gebruikt wordt, dus ziet het model elke epoch andere varianten. Precieze pixels kan het model nauwelijks uit het hoofd leren; het moet leren wat hetzelfde blijft.

        Waar dat gebeurt: in de `__getitem__` van een Dataset, zoals in [Datasets en DataLoaders](/course/open/torch-dataloader?lang=nl). De DataLoader vraagt elke epoch één keer om elk voorbeeld, dus krijgt elke epoch nieuwe willekeurige veranderingen. Alleen de training set krijgt ze: de validation set en de test set blijven zoals ze zijn, want die meten hoe het model het doet op echte cijfers.

        Voor de cijfers is de verandering hier een verschuiving. De cijfers zijn rijen van 784 getallen, en `.reshape(28, 28)` maakt er een afbeelding van, op een tensor net als op een NumPy-array. `torch.roll(img, shifts=(dy, dx), dims=(0, 1))` schuift een afbeelding van 28 bij 28 `dy` rijen omlaag en `dx` kolommen naar rechts; negatieve getallen gaan omhoog en naar links. Wat er aan de ene kant afvalt, komt aan de andere kant terug. Rond de meeste cijfers ligt een rand van leeg papier, dus dan rolt er alleen leeg papier rond. Niet rond allemaal: in 14 van de 1.000 trainingscijfers, vooral zevens, raakt de inkt de rand, en in ongeveer 1 op de 8 komt hij tot op twee pixels ervan. Een verschuiving naar die rand brengt dan een paar pixels inkt aan de andere kant terug. De verschuiving wordt getrokken met `random.randint`, zoals in [Code uit modules gebruiken](/course/open/py-imports?lang=nl).

        Niet bij elke verandering blijft het label kloppen. Een cijfer dat ondersteboven staat, kan een ander cijfer worden: een 6 wordt een 9. Kies veranderingen die echte voorbeelden ook hebben.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **data augmentation** | | nieuwe trainingsvoorbeelden van de voorbeelden die je hebt, met willekeurige veranderingen waarbij het label blijft kloppen | in `__getitem__` |
        | `torch.roll(t, shifts, dims)` | torch roll | schuift de getallen van een tensor langs de axes `dims`; wat er aan de ene kant afvalt, komt aan de andere kant terug | `torch.roll(img, shifts=(1, -1), dims=(0, 1))` |
        """),
        L.watch_out("Gebruik data augmentation alleen voor de training set. Een validation set met willekeurige veranderingen meet elke epoch iets anders, en een test set ermee zegt niet meer hoe het model het doet op de echte cijfers die het zal tegenkomen."),
        mo.md(r"""
        De figuur: trainingscijfer 0, een 5, zoals het is en op vier manieren één pixel verschoven, met de verschuiving (omlaag, opzij) eronder.
        """),
        mo.Html(C.variants_picture({"title": "Eén cijfer, vier verschuivingen", "original": "zoals het is",
                                    "alt": "Vijf afbeeldingen van dezelfde handgeschreven 5: zoals het is, en één pixel verschoven naar boven, beneden, links en rechts. Elk is duidelijk hetzelfde cijfer."})),
    ])
    return


@app.cell(hide_code=True)
def picture_curves():
    _epoch0, _loss0 = C.best(0)
    _epoch1, _loss1 = C.best(1)
    mo.vstack([
        mo.md(r"""
        Het effect: de validation loss per epoch van het MLP uit [Loss curves lezen](/course/open/torch-loss-curves?lang=nl), op dezelfde 1.000 trainingscijfers, zonder verschuivingen en met verschuivingen van hooguit 1 pixel. Van elk één run: in vier runs met andere seeds lag de laagste validation loss met verschuivingen tussen 0,25 en 0,29, en zonder verschuivingen telkens rond 0,41.
        """),
        mo.Html(C.curves_picture({"title": "Validation loss, met en zonder verschuivingen", "epoch": "epoch", "loss": "validation loss",
                                  "plain": "zonder verschuivingen", "shifted": "met verschuivingen",
                                  "alt": f"Twee gestippelde lijnen over 40 epochs. Zonder verschuivingen is de validation loss het laagst na epoch {_epoch0}, op {L.number(_loss0, 2)}, en stijgt hij daarna. Met verschuivingen daalt hij langer door, tot {L.number(_loss1, 2)} na epoch {_epoch1}, en blijft hij laag."})),
        mo.md(f"""
        Zonder verschuivingen is de validation loss het laagst na epoch {_epoch0}, op <span class="c-loss ea-num">{L.number(_loss0, 2)}</span>, en daarna stijgt hij. Met verschuivingen daalt hij door tot epoch {_epoch1}, naar <span class="c-loss ea-num">{L.number(_loss1, 2)}</span>. Leren duurt langer, omdat elke epoch nieuwe varianten brengt, maar het model wordt veel beter op cijfers die het nooit zag.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om te trainen met data augmentation:

    1. Kies veranderingen waarbij het label blijft kloppen, zoals een verschuiving van hooguit `k` pixels.
    2. Schrijf een Dataset waarvan `__getitem__` elke keer een nieuwe willekeurige verandering trekt, en die op het voorbeeld toepast.
    3. Gebruik die Dataset alleen voor de training set; de validation set en de test set blijven zoals ze zijn.
    4. Train zoals voorheen, en vergelijk de validation loss met een run zonder data augmentation.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een Dataset die elk cijfer hooguit 1 pixel verschuift, dezelfde als `C.ShiftedDigits` uit de figuur. Eerst geeft hij twee keer hetzelfde voorbeeld, om te laten zien dat de verschuiving elke keer nieuw is; `C.shift_of(x, original)` geeft de verschuiving die hij vindt, omlaag en opzij. Daarna traint de cel het MLP uit [Loss curves lezen](/course/open/torch-loss-curves?lang=nl) ermee, 40 epochs lang. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit; dat duurt een paar seconden.
    """)
    return


@app.cell
def example_augment():
    import random
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    class DigitsMLP(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.hidden = nn.Linear(784, hidden)
            self.out = nn.Linear(hidden, 10)

        def forward(self, x):
            return self.out(torch.relu(self.hidden(x)))

    # 1. de verandering: een verschuiving van hooguit k pixels, omlaag en opzij
    # 2. een Dataset die elke keer dat hij een voorbeeld geeft een nieuwe verschuiving trekt
    class ShiftedDigits(Dataset):
        def __init__(self, images, labels, k):
            self.images = images
            self.labels = labels
            self.k = k

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            img = self.images[i].reshape(28, 28)
            dy = random.randint(-self.k, self.k)
            dx = random.randint(-self.k, self.k)
            img = torch.roll(img, shifts=(dy, dx), dims=(0, 1))
            return img.reshape(784), self.labels[i]

    data = np.load("data/qmnist-small.npz")
    X = torch.tensor(data["train_images"], dtype=torch.float32) / 255
    y = torch.tensor(data["train_labels"], dtype=torch.int64)
    # 3. verschuivingen alleen voor de training set; de validation set blijft zoals hij is
    train_set = ShiftedDigits(X[:1000], y[:1000], 1)
    X_val, y_val = X[5000:5500], y[5000:5500]

    random.seed(0)
    print(C.shift_of(train_set[0][0], X[0]), C.shift_of(train_set[0][0], X[0]))

    # 4. train zoals voorheen, en meet na elke epoch de validation loss
    loss_fn = nn.CrossEntropyLoss()
    torch.manual_seed(0)
    random.seed(0)
    model = DigitsMLP(64)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    loader = DataLoader(train_set, batch_size=50, shuffle=True)
    torch.manual_seed(1)
    val_losses = []
    for epoch in range(40):
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_losses.append(loss_fn(model(X_val), y_val).item())
        model.train()
    best = np.argmin(val_losses)
    print(best + 1, val_losses[best], val_losses[-1])
    return


@app.cell(hide_code=True)
def text_after_example():
    _epoch1, _loss1 = C.best(1)
    _epoch0, _loss0 = C.best(0)
    mo.md(f"""
    Hetzelfde voorbeeld komt terug met twee verschillende verschuivingen: eerst (0, 0), een van de negen mogelijke verschuivingen, die het laat zoals het is, dan (−1, 0), één pixel omhoog. De validation loss is het laagst na epoch {_epoch1}, op {L.number(_loss1, 4)}, zoals in de figuur, tegen {L.number(_loss0, 2)} zonder verschuivingen. En na 40 epochs is hij nog steeds laag: het model overfit nauwelijks. De 1.000 trainingscijfers zijn niet veranderd; alleen de manier waarop het model ze ziet.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_where():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Welke set?
    """)
    return


@app.cell(hide_code=True)
def choice_where_cell():
    choice_where = L.choice(
        "where",
        question="Je traint met verschoven cijfers. Welke sets krijgen de willekeurige verschuivingen?",
        options={
            "all": "Alle drie: training, validation en test, zodat ze op elkaar lijken",
            "two": "De training set en de validation set",
            "right": "Alleen de training set",
        },
        correct="right",
        why={
            "all": "De test set zegt hoe het model het doet op echte cijfers; met willekeurige verschuivingen meet hij iets anders. Zie *Let op* in *Het idee*.",
            "two": "Een validation set met willekeurige verschuivingen geeft elke epoch een andere loss, ook voor hetzelfde model. Zie *Let op* in *Het idee*.",
            "right": "De verschuivingen geven het model meer gevarieerde voorbeelden om van te leren. De validation set en de test set meten, en blijven elke keer hetzelfde.",
        },
    )
    choice_where
    return (choice_where,)


@app.cell(hide_code=True)
def text_exercise_shifted():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een Dataset die verschuift

        `ShiftedDigits(images, labels, k)` geeft elk cijfer hooguit `k` pixels verschoven. `__getitem__` trekt al `dy`, de verschuiving omlaag. Vervang de `...`: trek `dx` op dezelfde manier, en verschuif `img` over `dy` rijen omlaag en `dx` kolommen opzij. De laatste regels vragen twee keer om voorbeeld 2, met `random.seed(3)` en `k` = 2, en laten zijn label zien en de twee verschuivingen die `shift_of` vindt, omlaag en opzij.
        """),
        L.task("Trek `dx`, en verschuif `img` met `torch.roll` over `dy` rijen omlaag en `dx` kolommen opzij."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shifted():
    box_shifted = L.code_box("shifted", lines=26)
    box_shifted
    return (box_shifted,)


@app.cell(hide_code=True)
def check_shifted(box_shifted):
    result_shifted = L.check("shifted", box_shifted.value)
    result_shifted
    return (result_shifted,)


@app.cell(hide_code=True)
def hints_shifted():
    L.hints(
        "shifted",
        [
            "Stap 2 van het recept, zoals in `__getitem__` van het uitgewerkte voorbeeld.",
            "`dx` komt van `random.randint` met dezelfde twee getallen als `dy`. Dan krijgt `torch.roll` `img`, de twee verschuivingen en de twee axes, rijen eerst.",
            "`dx = random.randint(-self.k, self.___)`, dan `img = torch.roll(img, shifts=(___, ___), dims=(0, 1))`.",
        ],
        why="Met `random.seed(3)` en `k` = 2 trekt de eerste aanroep een verschuiving van −1 omlaag, dus 1 omhoog, en 2 naar rechts; de tweede aanroep trekt 2 omlaag en 1 naar links. Het label blijft de 4 van voorbeeld 2.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_keeps():
    mo.md(r"""
    ### Oefening 3. Bij welke veranderingen blijft het hetzelfde cijfer?

    Zes veranderingen die je aan een trainingscijfer kunt doen. Zet elk in zijn groep: blijft hetzelfde cijfer, of kan het cijfer veranderen of onleesbaar maken.
    """)
    return


@app.cell(hide_code=True)
def exercise_keeps():
    sort_keeps = L.sort("keeps-label")
    sort_keeps
    return (sort_keeps,)


@app.cell(hide_code=True)
def check_keeps(sort_keeps):
    result_keeps = L.check("keeps-label", sort_keeps)
    result_keeps
    return (result_keeps,)


@app.cell(hide_code=True)
def hints_keeps():
    L.hints(
        "keeps-label",
        [
            "Vraag bij elke verandering: verschillen echte handgeschreven cijfers op deze manier van elkaar?",
            "Bij drie veranderingen blijft het hetzelfde cijfer, drie kunnen het breken. Denk aan een 2, een 3, een 6 en een 9.",
            "Bij een verschuiving, een kleine draai en lichtere inkt blijft het hetzelfde cijfer. Welke drie veranderingen kunnen van een 6 een 9 maken, van een 3 iets wat geen cijfer is, of van een 8 iets wat een 9 kan zijn?",
        ],
        why="**Blijft hetzelfde cijfer:** een verschuiving van een pixel, een draai van 10 graden, lichtere inkt. **Kan het veranderen of onleesbaar maken:** een spiegelbeeld, ondersteboven, en de onderste helft wegsnijden.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Data augmentation zit in de `__getitem__` uit [Datasets en DataLoaders](/course/open/torch-dataloader?lang=nl), en werkt op afbeeldingen als de roosters van getallen uit [Een afbeelding is een rooster van getallen](/course/open/data-images-as-arrays?lang=nl). De overfitting uit [Loss curves lezen](/course/open/torch-loss-curves?lang=nl) komt later en is kleiner, omdat het model geen precieze pixels meer uit zijn hoofd kan leren. In de volgende les, *Een pretrained model gebruiken*, train je zelf helemaal geen cijferlezer: je laadt er een die iemand anders getraind heeft.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_where, result_keeps, result_shifted):
    L.wrap_up(choice_where, result_shifted, result_keeps)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
