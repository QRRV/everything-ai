# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nn-trainer. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-trainer", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Learning-rate schedules](/course/open/opt-lr-schedules?lang=nl) en [Gradient clipping](/course/open/opt-gradient-clipping?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sched():
    quick_sched = L.choice(
        "quick-sched",
        question="Een schedule die in stappen telt, zet de learning rate van elke stap. Wanneer roep je `scheduler.step()` aan?",
        options={
            "epoch": "Eén keer aan het eind van elke epoch, hoeveel stappen die ook had",
            "right": "Na elke `opt.step()`, zodat de volgende stap zijn learning rate krijgt",
            "forward": "Vóór elke forward pass, zodat de loss de nieuwe learning rate gebruikt",
        },
        correct="right",
        why={
            "epoch": "Een schedule die in stappen telt, schuift dan één stap op per epoch, en zijn warmup en daling duren veel te lang. Zie [Learning-rate schedules](/course/open/opt-lr-schedules?lang=nl).",
            "forward": "De loss gebruikt de learning rate niet; alleen de stap van de optimizer doet dat. Zie [Learning-rate schedules](/course/open/opt-lr-schedules?lang=nl).",
            "right": "Eén stap van de optimizer, één stap van de schedule. Vandaag gaat dat paar in een loop die je één keer schrijft, met één verschil: niet elke mini-batch krijgt een stap.",
        },
    )
    quick_sched
    return (quick_sched,)


@app.cell(hide_code=True)
def quick_check_clip():
    quick_clip = L.choice(
        "quick-clip",
        question="Je clipt op een norm van 1. De ene stap heeft een gradient met norm 0,5, de volgende norm 3. Wat gebeurt er met die twee?",
        options={
            "both": "Allebei worden ze geschaald naar norm 1, zodat elke stap even groot is",
            "half": "Allebei worden ze gehalveerd, omdat de grens 1 is",
            "right": "De eerste blijft zoals hij is, de tweede gaat omlaag naar norm 1",
        },
        correct="right",
        why={
            "both": "Clippen maakt een gradient alleen kleiner; een gradient onder de grens blijft zoals hij is. Zie [Gradient clipping](/course/open/opt-gradient-clipping?lang=nl).",
            "half": "Clippen schaalt met `max_norm / norm`, en alleen als de norm boven de grens ligt. Zie [Gradient clipping](/course/open/opt-gradient-clipping?lang=nl).",
            "right": "Onder de grens verandert er niets; erboven houdt de gradient zijn richting en krijgt hij norm 1. Vandaag wordt clippen één regel van een loop die elk project gebruikt.",
        },
    )
    quick_clip
    return (quick_clip,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Elk project tot nu toe had zijn eigen training loop, en elk nieuw idee voegde er regels aan toe: evalueren, een logboek, checkpoints, clippen, een schedule. Kopieer je zo'n loop naar elk project, dan kopieer je ook zijn fouten, en een fix in de ene kopie ontbreekt in de andere. Je schrijft hem één keer, test hem één keer en importeert hem overal. En er komt één nieuw idee bij: op een kleine GPU laat een groot model alleen ruimte voor een kleine mini-batch, terwijl trainen beter gaat met een grote. **Gradient accumulation** telt de gradients van een paar kleine mini-batches bij elkaar op vóór één stap.
    """)
    return


@app.cell(hide_code=True)
def predict_accum():
    guess_accum = L.guess(
        "four-small",
        question="Een model krijgt één mini-batch van 64 voorbeelden, met `backward()` van de gemiddelde loss. Een andere kopie van hetzelfde model krijgt dezelfde 64 voorbeelden als vier mini-batches van 16, met `backward()` van elke gemiddelde loss gedeeld door 4, en geen stap of `zero_grad()` ertussen. Hoe verhouden de twee gradients zich?",
        options={
            "four": "De tweede is ongeveer vier keer zo groot",
            "right": "Ze zijn gelijk, op afronding na",
            "quarter": "De tweede is een kwart zo groot",
        },
        reveal="**Gelijk, op afronding na.** `backward()` telt elke nieuwe gradient op bij die er al is. Elk van de vier gemiddelde losses is het gemiddelde over 16 voorbeelden; een kwart van elk, bij elkaar opgeteld, is het gemiddelde over alle 64. In het uitgewerkte voorbeeld hieronder is het grootste verschil kleiner dan 0,000001.",
    )
    guess_accum
    return (guess_accum,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Gradient accumulation** met $k$ mini-batches van $B$ voorbeelden: per mini-batch `backward()` van zijn gemiddelde loss gedeeld door $k$; na elke $k$ mini-batches één stap. De gradients tellen op, dus de stap is die van één mini-batch van $kB$ voorbeelden, de **effectieve batch size**:
        """),
        L.formula(r"[[grad:\nabla \mathcal{L}_{kB}]] = \frac{1}{[[hyper:k]]} \sum_{j=1}^{[[hyper:k]]} [[grad:\nabla \mathcal{L}_{B,\,j}]]"),
        mo.md(r"""
        Een **trainer** is de loop eromheen, één keer geschreven: train elke epoch met accumulation, clippen en een schedule; meet daarna de loss op de validation set; schrijf een regel in het logboek; en sla de parameters op als de validation loss de laagste tot nu toe is. Alles wat hij nodig heeft, krijgt hij als argument: het model, de twee loaders, het aantal epochs, de piek van de learning rate, de warmup in stappen van de optimizer, `max_norm`, $k$ en waar hij moet opslaan. Hier gebruikt hij AdamW, en `torch.optim.lr_scheduler.LambdaLR`, die de piek van de learning rate bij elke stap vermenigvuldigt met wat een functie van de stap geeft: de factor met warmup en cosine decay uit [Learning-rate schedules](/course/open/opt-lr-schedules?lang=nl).

        De regel met de loss is zo geschreven dat hij ook past bij een language model, waarvan de logits een tijdas hebben: shape (batch, tijd, vocabulary). `logits.reshape(-1, logits.shape[-1])` zet alle voorspellingen onder elkaar: de `-1` betekent zoveel rijen als nodig is, en `yb.reshape(-1)` hun targets, zodat `F.cross_entropy` voor beide soorten model de shapes krijgt die hij wil.

        De oefeningen gebruiken drie nieuwe stukjes Python; die staan ook in de tabel.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **gradient accumulation** | | de gradients van een paar mini-batches bij elkaar optellen vóór één stap | `(loss / accum).backward()` |
        | **effectieve batch size** | | het aantal voorbeelden per stap: de grootte van een mini-batch keer $k$ | `batch_size * accum` |
        | $k$ | k | het aantal mini-batches per stap | `accum` |
        | **trainer** | | een training loop die je één keer schrijft, met evalueren, een logboek, checkpoints, clippen en een schedule | `Trainer(...).fit()` |
        | `%` | modulo | de rest na delen: `7 % 3` is 1, dus `k % 4 == 0` is `True` als `k` een veelvoud is van 4 | `k % accum == 0` |
        | `or` | | `True` als de voorwaarde links of die rechts ervan `True` is, of allebei | `k % accum == 0 or k == len(loader)` |
        | `is None` | | `True` als een variabele `None` is; `is not None` is het omgekeerde: er is iets | `if sched is not None:` |
        """),
        L.watch_out("Met gradient accumulation telt de schedule stappen, geen mini-batches: een epoch van 94 mini-batches met $k = 4$ heeft 24 stappen, de laatste al na 2 mini-batches. En een batch normalization layer ziet nog steeds alleen de kleine mini-batch: accumulation laat zijn statistieken niet uit meer voorbeelden komen."),
        mo.md(r"""
        De figuur: de trainer op het MLP uit [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl), 8 epochs AdamW met een piek van de learning rate van 0,003, warmup in de eerste epoch en cosine decay naar 0, en clippen op 1. Links het logboek: de training loss is het gemiddelde over de epoch, dus in epoch 1 ligt hij boven de validation loss, die aan het eind gemeten is; daarna dalen ze allebei, de validation loss van 0,50 naar 0,27. Rechts de learning rate bij de start en aan het eind van elke epoch. Na epoch 8 leest het netwerk 915 van de 1.000 testcijfers goed.
        """),
        mo.Html(C.history_picture({"loss_title": "Loss na elke epoch", "lr_title": "Learning rate", "x": "epoch", "loss": "loss", "lr": "learning rate",
                                   "train": "training", "val": "validation",
                                   "alt": "Links: de training loss na elke epoch daalt van 1,25 naar 0,11, de validation loss van 0,50 naar 0,27. Rechts: de learning rate, bijna 0 bij de start, 0,003 na de warmup van epoch 1, daarna dalend langs een cosinus naar 0 na epoch 8."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een trainer, epoch na epoch:

    1. Voor elke mini-batch: zijn loss, gedeeld door $k$, en `backward()`; de gradients tellen op.
    2. Na elke $k$ mini-batches, en na de laatste: clippen, de stap, de gradients op nul, en de schedule een stap verder.
    3. Na elke epoch: de gemiddelde loss op de validation set, in eval mode en zonder graaf, en terug naar train mode.
    4. Een regel in het logboek met de epoch, beide losses en de learning rate; sla de parameters op als de validation loss de laagste tot nu toe is.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 1, gecontroleerd: de gradient van één mini-batch van 64 tegen die van vier mini-batches van 16, voor een `nn.Linear(4, 3)` met willekeurige data. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 1 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_accumulation():
    import torch
    import torch.nn.functional as F
    from torch import nn

    torch.manual_seed(0)
    _model = nn.Linear(4, 3)
    _X = torch.randn(64, 4)
    _y = (_X[:, 0] > 0).long() + (_X[:, 1] > 0).long()
    # om mee te vergelijken: de gradient van één mini-batch van 64
    F.cross_entropy(_model(_X), _y).backward()
    _full = _model.weight.grad
    _model.zero_grad()
    # 1. vier mini-batches van 16: voor elk backward, zonder stap of zero_grad ertussen
    for _k in range(4):
        _loss = F.cross_entropy(_model(_X[16 * _k:16 * _k + 16]), _y[16 * _k:16 * _k + 16])
        # maak af: deel de loss eerst door 4
        _loss.backward()
    print([round(v, 4) for v in _full[0].tolist()])
    print([round(v, 4) for v in _model.weight.grad[0].tolist()])
    print(round((_model.weight.grad - _full).abs().max().item(), 6))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De eerste regel is rij 0 van de gradient van de mini-batch van 64: 0,2481, 0,118, −0,0296 en −0,0886. `_model.zero_grad()` maakt de gradient leeg voor de tweede manier, en laat `_full` zoals hij was. Zoals hij is, telt elke `backward()` van de vier zijn hele gradient op: rij 0 is 0,9922, 0,4718, −0,1185 en −0,3546, vier keer zo groot, en het grootste verschil is 0,744157. Afgemaakt, met `(_loss / 4).backward()`, zijn de rijen gelijk en print het grootste verschil `0.0`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_divide():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Waarom delen door k?
    """)
    return


@app.cell(hide_code=True)
def choice_divide_cell():
    choice_divide = L.choice(
        "why-divide",
        question="Waarom deel je bij gradient accumulation elke loss door $k$ vóór `backward()`?",
        options={
            "right": "Zodat de opgetelde gradient een gemiddelde is, zoals bij één grote mini-batch",
            "memory": "Om geheugen te sparen: een kleinere loss neemt minder ruimte in de graaf in",
            "clip": "Zodat de gradient altijd onder de grens van het clippen blijft, wat er ook gebeurt",
        },
        correct="right",
        why={
            "memory": "Een loss is hoe dan ook één getal. Wat geheugen spaart, is de kleine mini-batch zelf.",
            "clip": "Delen maakt de gradient k keer kleiner dan zonder, niet altijd klein; clippen heeft nog steeds zijn eigen taak.",
            "right": "`backward()` telt op. Zonder delen geven k mini-batches k keer de gradient van een gemiddelde, en met SGD is de stap k keer groter; met de 1/k is de stap gelijk aan die van één mini-batch met k keer zoveel voorbeelden.",
        },
    )
    choice_divide
    return (choice_divide,)


@app.cell(hide_code=True)
def text_exercise_epoch():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Eén epoch, met alles erop en eraan

        Maak `train_epoch(model, loader, opt, sched, max_norm, accum)` af: één epoch volgens stap 1 en 2 van het recept. De regel met de loss staat er, en `k` telt de mini-batches. Vervang de `...`: `backward()` van de loss gedeeld door `accum`; en als `k` een veelvoud is van `accum`, of de laatste mini-batch (`k == len(loader)`): clip als `max_norm` niet `None` is, doe de stap, zet de gradients op nul en laat de schedule een stap verder gaan als `sched` niet `None` is. De check draait een epoch van 5 mini-batches met `accum = 2`, dan hetzelfde met clippen en een schedule, en dan een model met logits van shape (4, 3, 5), zoals een language model ze geeft. Als hij goed is, slaat de check hem op in je project, als `work/projects/nn/train_epoch.py`.
        """),
        L.task("Laat `train_epoch` één epoch trainen met gradient accumulation, clippen en een schedule."),
    ])
    return


@app.cell(hide_code=True)
def exercise_epoch():
    box_epoch = L.code_box("train-epoch", lines=26)
    box_epoch
    return (box_epoch,)


@app.cell(hide_code=True)
def check_epoch(box_epoch):
    result_epoch = L.check("train-epoch", box_epoch.value)
    result_epoch
    return (result_epoch,)


@app.cell(hide_code=True)
def hints_epoch():
    L.hints(
        "train-epoch",
        [
            "Eerst de regel uit het afgemaakte uitgewerkte voorbeeld, met `accum` in plaats van 4. Dan één `if` met `%` en `or`, en daaronder, in deze volgorde: clippen, `opt.step()`, `opt.zero_grad()` en de schedule.",
            "Clippen en de schedule zijn optioneel: elk krijgt zijn eigen `if ... is not None:`, met de regel uit [Gradient clipping](/course/open/opt-gradient-clipping?lang=nl) en `sched.step()`.",
        ],
        why="Door te delen door `accum` tellen de gradients van de mini-batches op tot een gemiddelde. De stap komt na elke `accum` mini-batches, en na de laatste, zodat er geen gradient verloren gaat. Clippen komt vóór de stap, omdat de stap de gradients gebruikt, en de schedule erna, één keer per stap van de optimizer.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_trainer(result_epoch):
    result_epoch   # opnieuw gemaakt na de check van oefening 2, die je train_epoch opslaat
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De trainer

        Maak de class `Trainer` af. `__init__` staat er: die bewaart alles, maakt AdamW en de schedule, en een lege `self.history`; `train_epoch` komt uit je project. Vervang de twee `...`:

        - `evaluate()` geeft de gemiddelde loss op `self.val_loader`, volgens stap 3: `self.model.eval()`, een loop onder `torch.no_grad()` met dezelfde regel voor de loss als in `train_epoch`, en aan het eind `self.model.train()`.
        - `fit()` doet voor elke epoch stap 4: `train_epoch` met de optimizer, de schedule, `max_norm` en `accum` van de trainer; dan `evaluate()`; dan een dict in `self.history` met `"epoch"` (vanaf 1), `"train_loss"`, `"val_loss"` en `"lr"`, de learning rate van `self.opt.param_groups[0]`; en `torch.save(self.model.state_dict(), self.path)` als de validation loss de laagste tot nu toe is en `self.path` niet `None` is.

        De check traint kleine netwerken op willekeurige data: hij vergelijkt het logboek, laadt de opgeslagen parameters, en meet twee keer met dropout. Als hij goed is, slaat de check hem op als `work/projects/nn/trainer.py`.
        """),
        L.course_versions(),
        L.task("Laat `Trainer` epoch na epoch evalueren, een logboek bijhouden en de beste parameters opslaan."),
    ])
    return


@app.cell(hide_code=True)
def exercise_trainer():
    box_trainer = L.code_box("trainer", lines=44)
    box_trainer
    return (box_trainer,)


@app.cell(hide_code=True)
def check_trainer(box_trainer):
    result_trainer = L.check("trainer", box_trainer.value)
    result_trainer
    return (result_trainer,)


@app.cell(hide_code=True)
def hints_trainer():
    L.hints(
        "trainer",
        [
            "`evaluate` is de loop uit [Loss curves lezen](/course/open/torch-loss-curves?lang=nl) voor de validation set, met de regel voor de loss uit `train_epoch`; tel `.item()` van elke loss op bij een totaal en deel door `len(self.val_loader)`.",
            "In `fit` begint `best` bij `math.inf`, oneindig, boven elke loss. Een dict schrijf je als `{\"epoch\": epoch + 1, ...}`; het opslaan komt onder `if val_loss < best:`, waaronder je ook `best = val_loss` zet.",
        ],
        why="`evaluate` zet eval mode aan, zodat dropout uit staat tijdens het meten, en aan het eind weer train mode, anders zou de volgende epoch zonder dropout trainen; `torch.no_grad()` bouwt geen graaf, wat tijd en geheugen spaart. `fit` leest de learning rate na `train_epoch`, dus het logboek heeft de learning rate aan het eind van de epoch, zoals in de figuur, en het slaat alleen op als de validation loss lager is dan elke eerdere.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De loop uit [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl) groeit uit tot een trainer, met de validation loss uit [Loss curves lezen](/course/open/torch-loss-curves?lang=nl), het beste checkpoint uit [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl), het clippen uit [Gradient clipping](/course/open/opt-gradient-clipping?lang=nl), en AdamW uit [AdamW: weight decay zoals het hoort](/course/open/opt-adamw?lang=nl) met de warmup en cosine decay uit [Learning-rate schedules](/course/open/opt-lr-schedules?lang=nl). Vanaf nu importeert een project hem: `from nn.trainer import Trainer`. Grotere libraries, zoals PyTorch Lightning, hebben hetzelfde soort trainer, met callbacks: functies die je hem meegeeft, en die hij op vaste momenten aanroept, zoals na elke epoch. In de volgende les, *Project: een diep netwerk laten trainen*, traint een diep MLP eerst niet, en krijg je het stap voor stap aan het trainen, met deze trainer.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_divide, result_epoch, result_trainer):
    L.wrap_up(choice_divide, result_epoch, result_trainer)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
