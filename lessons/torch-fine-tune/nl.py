# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les torch-fine-tune. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("torch-fine-tune", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een pretrained netwerk als feature extractor](/course/open/torch-frozen-features?lang=nl) en [Te klein, te groot: de learning rate](/course/open/ml-learning-rate?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_frozen():
    quick_frozen = L.choice(
        "quick-frozen",
        question="Met 100 gelabelde cijfers van de oude scanner las de bevroren cijferlezer met een nieuwe laatste layer er 860 tot 883 van de 1.000 goed, een nieuw netwerk vanaf nul 629 tot 669. Waar komt dat verschil vandaan?",
        options={
            "longer": "Hij trainde langer: meer epochs dan het netwerk vanaf nul",
            "right": "Er leren maar 650 getallen, op features die al cijfers beschrijven",
            "bigger": "Hij heeft meer parameters, dus hij kan meer leren",
        },
        correct="right",
        why={
            "longer": "Andersom: 30 epochs voor de nieuwe laatste layer, 60 voor het netwerk vanaf nul. Zie [Een pretrained netwerk als feature extractor](/course/open/torch-frozen-features?lang=nl).",
            "bigger": "Beide hebben dezelfde layers, en dus evenveel parameters; in de bevroren cijferlezer leren er maar 650 van. Zie [Een pretrained netwerk als feature extractor](/course/open/torch-frozen-features?lang=nl).",
            "right": "Vanaf nul leren 52.000 getallen van 100 afbeeldingen, en die leren ze uit het hoofd. In deze les leren de geleende layers ook, al is het maar een beetje.",
        },
    )
    quick_frozen
    return (quick_frozen,)


@app.cell(hide_code=True)
def quick_check_rate():
    quick_rate = L.choice(
        "quick-big-lr",
        question="Een loss curve springt op en neer, of stijgt. Wat betekent dat?",
        options={
            "small": "De learning rate is te klein: de stappen zijn te kort",
            "few": "De training data heeft te weinig voorbeelden",
            "right": "De learning rate is te groot: elke stap gaat te ver",
        },
        correct="right",
        why={
            "small": "Te korte stappen geven een curve die langzaam daalt, maar wel daalt. Zie [Te klein, te groot: de learning rate](/course/open/ml-learning-rate?lang=nl).",
            "few": "Te weinig voorbeelden geven een training loss die daalt terwijl de validation loss hoog blijft: overfitting. Zie [Loss curves lezen](/course/open/torch-loss-curves?lang=nl).",
            "right": "Elke stap springt voorbij de bodem van het dal. In deze les doet een stap die te ver gaat nog een ander soort schade: hij brengt geleende features ver weg van wat ze wisten.",
        },
    )
    quick_rate
    return (quick_rate,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De bevroren cijferlezer met een nieuwe laatste layer leest 871 van de 1.000 gespikkelde testcijfers goed. Zijn features bleven zoals ze waren: geleerd op 6.000 schone cijfers, en ze zagen nooit een spikkel. Laat ze zich aanpassen aan de cijfers van de oude scanner, dan leest hij er meer. Maar niet veel meer: met 100 voorbeelden leren layers die veel veranderen die 100 uit het hoofd, zoals een netwerk vanaf nul doet.

    **Een voorbeeld uit het dagelijks leven.** Een confectiepak dat bijna past. Een kleermaker naait geen nieuw pak: eerst kies je de maat die het best past, dan neemt de kleermaker de mouwen en de taille een beetje in. Knip je er te veel af, dan is het pak verpest.
    """)
    return


@app.cell(hide_code=True)
def predict_skip():
    guess_skip = L.guess(
        "skip-head",
        question="Fine-tuning heeft twee fases: eerst leert alleen de nieuwe laatste layer, dan elke layer, met een kleine learning rate. Wat als je fase 1 overslaat: een nieuwe laatste layer, elke layer meteen ontdooid, en dan 30 epochs met de kleine learning rate 0,001 op dezelfde 100 cijfers? Fase 1 alleen gaf 860 tot 883 van de 1.000.",
        options={
            "right": "Minder: de geleende layers volgen de fouten van een willekeurige nieuwe layer",
            "more": "Meer: elke layer past zich vanaf de allereerste stap aan de spikkels aan, niet alleen de laatste",
            "same": "Ongeveer evenveel: dezelfde cijfers, en dezelfde kleine learning rate",
        },
        reveal=f"**Minder: {C.REFERENCE['no-head'][0]} tot {C.REFERENCE['no-head'][1]}**, over 5 seeds, bij elke seed minder dan fase 1 alleen; met 60 epochs {C.REFERENCE['no-head-60'][0]} tot {C.REFERENCE['no-head-60'][1]}. Een nieuwe laatste layer begint met willekeurige weights, dus eerst zeggen zijn fouten niets over cijfers. Hun gradients komen ook bij de geleende layers: de eerste stappen schuiven de features op naar wat bij een willekeurige layer past, en na 30 epochs zijn de geleende weights ongeveer twee keer zo ver verschoven als wanneer fase 1 eerst komt. Met eerst fase 1: {C.REFERENCE['tuned'][0]} tot {C.REFERENCE['tuned'][1]}.",
    )
    guess_skip
    return (guess_skip,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Twee fases.** Begin waar de vorige les eindigde: de cijferlezer bevroren, met een nieuwe laatste layer getraind op jouw voorbeelden. Dat is fase 1. Ontdooi dan elke parameter, `p.requires_grad = True`, en train ze allemaal door, met een learning rate die ongeveer tien keer zo klein is: fase 2. De geleende features schuiven een beetje op naar jouw data. Dat is **fine-tuning**.

        **Waarom eerst fase 1.** In fase 2 leert elke layer van de fouten van de laatste layer. Na fase 1 gaan die fouten over de gespikkelde cijfers, dus leren de geleende layers wat helpt om ze te lezen. Een laatste layer die nog willekeurig is, maakt fouten die niets zeggen, en die zouden de geleende layers volgen.

        **Waarom een kleine learning rate.** De geleende features zijn al goed: ze hoeven maar een beetje te veranderen. Grote stappen brengen ze ver weg van wat 6.000 cijfers ze leerden, naar wat alleen bij deze 100 past.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **fine-tuning** | | alle layers van een pretrained netwerk verder trainen op de data van een nieuwe taak, met een kleine learning rate | `p.requires_grad = True`, dan trainen |
        """),
        L.watch_out("Alleen ontdooien is niet genoeg: `train` moet ook elke parameter krijgen, `model.parameters()`. Geef je alleen `model.out.parameters()`, dan verandert fase 2 weer alleen de laatste layer."),
        mo.md(r"""
        De figuur: de cijferlezer in de twee fases.
        """),
        mo.Html(C.phases_picture({"title": "De twee fases van fine-tuning", "phase1": "fase 1", "rate1": "learning rate 0,01",
                                  "phase2": "fase 2", "rate2": "learning rate 0,001", "epochs": "30 epochs", "learns": "leert", "frozen": "bevroren",
                                  "alt": "Twee kolommen met de vier layers conv1, conv2, hidden en out. In fase 1, met een learning rate van 0,01, zijn conv1, conv2 en hidden bevroren en leert alleen out. In fase 2, met een learning rate van 0,001, leren ze alle vier."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een pretrained netwerk te finetunen voor een nieuwe taak:

    1. Fase 1, zoals in [Een pretrained netwerk als feature extractor](/course/open/torch-frozen-features?lang=nl): bevries elke parameter, geef het netwerk een nieuwe laatste layer, en train alleen die layer.
    2. Ontdooi elke parameter: `for p in model.parameters(): p.requires_grad = True`.
    3. Fase 2: train alle parameters door, met een learning rate die ongeveer tien keer zo klein is als in fase 1.
    4. Test op voorbeelden waarop het niet trainde, en vergelijk met fase 1 alleen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De vier stappen voor de 10 soorten kleding van Fashion-MNIST, uit de eerste 100 trainingsafbeeldingen, zoals in oefening 3 van de vorige les: daar las fase 1 alleen 506 tot 539 van de 1.000 testafbeeldingen goed, over 5 seeds, en een nieuw netwerk vanaf nul 617 tot 713. `C.train(model, params, X, y, lr, epochs)` is de training loop van de course uit de vorige les.
    """)
    return


@app.cell
def example_clothes():
    import numpy as np
    import torch
    from torch import nn
    from everything_ai.models import digit_cnn

    _data = np.load("data/fashion-mnist-small.npz")
    _X = torch.tensor(_data["train_images"][:100], dtype=torch.float32) / 255
    _y = torch.tensor(_data["train_labels"][:100])
    _X_test = torch.tensor(_data["test_images"], dtype=torch.float32) / 255
    _y_test = torch.tensor(_data["test_labels"])
    # 1. fase 1: de cijferlezer bevroren, een nieuwe laatste layer, en alleen die layer leert
    torch.manual_seed(0)
    _model = digit_cnn()
    for _p in _model.parameters():
        _p.requires_grad = False
    _model.embed = nn.Identity()
    _model.out = nn.Linear(64, 10)
    C.train(_model, _model.out.parameters(), _X, _y, 0.01, 30)
    print("na fase 1:", C.right(_model, _X_test, _y_test), "van de", len(_y_test), "goed")
    # 2. ontdooi elke parameter
    for _p in _model.parameters():
        _p.requires_grad = True
    for _name, _p in _model.named_parameters():
        print(_name, _p.requires_grad)
    # 3. fase 2: elke parameter leert, met een learning rate die tien keer zo klein is
    C.train(_model, _model.parameters(), _X, _y, 0.001, 30)
    # 4. test opnieuw
    print("na fase 2:", C.right(_model, _X_test, _y_test), "van de", len(_y_test), "goed")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Fase 2 schoof de features van cijfers op naar kleding: 586 goed in plaats van 533. Over 5 seeds: 566 tot 593 in plaats van 506 tot 539, bij elke seed meer. Toch doet een nieuw netwerk vanaf nul het beter met deze 100 afbeeldingen: 617 tot 713. In 30 epochs met kleine stappen worden features van cijfers geen features van kleding. Fine-tuning past features aan die bijna passen; het vervangt geen features die niet passen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_tune():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De twee fases

        Maak `fine_tune(model, train)` af, voor de 100 gelabelde cijfers van de oude scanner: stap 1 tot en met 3 van het recept. `model` is de cijferlezer zoals de vorige les hem achterlaat: bevroren, met een nieuwe laatste layer van 64 naar 10. `train(params, lr, epochs)` is `C.train` met het model en de 100 cijfers al ingevuld: hij traint de parameters `params` van `model`, met learning rate `lr`, `epochs` epochs lang. Hij krijgt dus drie waarden, geen zes. De check roept `fine_tune(reader, train)` aan met een `train` die niets traint, maar per aanroep noteert welke parameters hij zou veranderen, met welke learning rate en hoeveel epochs.
        """),
        L.task("Laat `fine_tune` eerst alleen de nieuwe laatste layer trainen, en dan elke layer met een learning rate die tien keer zo klein is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tune():
    box_tune = L.code_box("fine-tune", lines=9)
    box_tune
    return (box_tune,)


@app.cell(hide_code=True)
def check_tune(box_tune):
    result_tune = L.check("fine-tune", box_tune.value)
    result_tune
    return (result_tune,)


@app.cell(hide_code=True)
def hints_tune():
    L.hints(
        "fine-tune",
        [
            "Stap 1 tot en met 3 van het uitgewerkte voorbeeld doen hetzelfde voor kleding, met `C.train`. Hier kent `train` het model en de data al: hij krijgt alleen de parameters, de learning rate en de epochs.",
            "Fase 1 geeft `train` de parameters van de nieuwe layer, `model.out.parameters()`; fase 2 geeft hem elke parameter, `model.parameters()`. In de loop is ontdooien bevriezen andersom.",
            "`train(model.out.parameters(), 0.01, 30)`, dan `p.requires_grad = True` in de loop, dan `train(..., 0.001, 30)` met elke parameter.",
        ],
        why="Fase 1 traint de nieuwe laatste layer op de bevroren features, tot zijn fouten iets zeggen over de gespikkelde cijfers. De loop laat elke parameter weer leren, en fase 2 geeft ze allemaal aan `train`, met stappen die tien keer zo klein zijn, zodat de geleende features maar een beetje veranderen.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Jouw twee fases op de cijfers van de oude scanner

    De cel hieronder geeft de echte training loop van de course, op de 100 gelabelde cijfers, aan jouw `fine_tune` zodra die klopt, anders aan die van de course, en test het resultaat op de 1.000 gespikkelde testcijfers, naast fase 1 alleen. Dat duurt een paar seconden. Terwijl elke run traint, toont een grafiek zijn loss op de 100 trainingscijfers na elke epoch, en gestreept op de 1.000 testcijfers bij de epochs waar je naar terug kunt spoelen: eerst van fase 1 alleen, dan van de twee fases van `fine_tune` na elkaar, met de test loss van fase 1 alleen dun en gestreept ernaast. Naast of onder de grafiek staat wat de cijferlezer op dat moment gokt bij één gespikkeld testcijfer van elke soort, en daaronder wat zijn layer `conv2` maakt van een gespikkelde 9; allebei worden ze opnieuw getekend terwijl hij traint. Daarna kun je door beide fases terugspoelen.
    """)
    return


@app.cell(hide_code=True)
def demo_tune(box_tune, result_tune):
    _now = {}   # the live chart of each run, made at its first report, and its test loss so far
    _heads = {"frozen": {"guesses": "De cijferlezer: wat hij gokt bij één gespikkeld testcijfer van elke soort",
                         "layer": "Wat zijn layer {name} maakt van een gespikkelde 9: bevroren, dus na elke epoch hetzelfde"},
              "tuned": {"guesses": "De cijferlezer: wat hij gokt bij één gespikkeld testcijfer van elke soort",
                        "layer": "Wat zijn layer {name} maakt van een gespikkelde 9: in fase 1 bevroren, in fase 2 leert hij mee"}}

    def _progress(_p):
        _run = _p["run"]
        if _run not in _now or _p["epoch"] == 1:   # a new run, also when the course's runs again in place of yours
            _tu = _run == "tuned"
            _now[_run] = L.live({"title": "Beide fases" if _tu else "Fase 1 alleen", "x": "epoch", "y": "loss",
                                 "alt": "De loss van beide fases tot nu toe: op de 100 trainingscijfers na elke epoch, en gestreept op de 1.000 testcijfers bij de epochs van het terugspoelen; dun en gestreept de test loss van fase 1 alleen." if _tu else "De loss in fase 1 alleen tot nu toe: op de 100 trainingscijfers na elke epoch, en gestreept op de 1.000 testcijfers bij de epochs van het terugspoelen."}, _p["epochs"], low=0.0,
                                key=[("loss", False, "100 trainingscijfers, elke epoch"), ("loss", True, "1.000 testcijfers, bij de epochs van het terugspoelen")])
            _now[_run].watch(lambda: C.now_pictures(_p["look"](), _heads[_run], L.lang, f"nv-tune-{_run}-now"))
            _now[_run + "-test"] = []
            if _tu and "frozen-test" in _now:   # the test loss of phase 1 alone, thin and dashed
                _now["tuned"].reference(_now["frozen-test"], "fase 1 alleen", role="ink-2")
        _who = "Beide fases" if _run == "tuned" else "Fase 1 alleen"
        _line = f"**{_who}** · fase {_p['phase']} · epoch {_p['epoch']} van {_p['epochs']} · loss {L.number(_p['loss'], 3)}"
        if "val_loss" in _p:   # the test digits at the epochs of the rewind, dashed
            _now[_run + "-test"].append((_p["epoch"], _p["val_loss"]))
            _now[_run].reference(_now[_run + "-test"], "", role="loss", replace=True)
            _line += f" · test loss {L.number(_p['val_loss'], 3)}"
        _now[_run].add(_p["epoch"], _p["loss"], _line)

    _d = C.demo(box_tune.value if result_tune.state == "correct" else None, _progress)
    _who = "jouw `fine_tune`" if _d["yours"] else "de `fine_tune` van de course"
    _labels = {"title": "Loss op de testcijfers", "x": "epoch", "y": "test loss", "frozen": "fase 1 alleen", "alt": "De loss op de 1.000 testcijfers door beide fases heen, bij de epochs van het terugspoelen, en dun en gestreept die van fase 1 alleen."}
    _when = [_e for _e, _ in _d["tests"]["tuned"]] if len(_d["looked"]["tuned"]) == len(_d["tests"]["tuned"]) and all(_d["looked"]["tuned"]) else []

    def _frame(_e):
        _e = int(_e)
        _i = _when.index(_e)
        _k = _d["phases"]["tuned"][_i]
        _right = sum(_g == _t for _g, _t in zip(_d["looked"]["tuned"][_i], _d["truth"]))
        return (C.epoch_frame(_d, _e, _labels, _heads["tuned"], L.lang),
                f"Na epoch {_e}, in fase {_k}: van deze 10 testcijfers leest de cijferlezer er {_right} goed.")

    _table = mo.md(f"""Met {_who}, van de 1.000 gespikkelde testcijfers:

| | goed gelezen |
|---|---|
| fase 1 alleen: bevroren, met een nieuwe laatste layer | {_d['frozen']} |
| beide fases: na fine-tuning | {_d['tuned']} |

Over 5 seeds, met de code van de course: {C.REFERENCE['frozen'][0]} tot {C.REFERENCE['frozen'][1]} na fase 1 alleen, {C.REFERENCE['tuned'][0]} tot {C.REFERENCE['tuned'][1]} na fine-tuning. Bij elke seed leest fine-tuning er {C.REFERENCE['gain'][0]} tot {C.REFERENCE['gain'][1]} cijfers meer goed: een kleine winst, bovenop wat de geleende features al gaven.""")
    if len(_when) > 1:   # rewinding through both phases
        _out = mo.vstack([L.explore("rewind", "epoch", _frame, start=_when[-1], values=_when, role="",
                                    task="Ga door fase 1 en verder in fase 2: wanneer begint layer `conv2` te veranderen, en veranderen de gokken mee?"), _table])
    else:
        _out = _table
    _out
    return


@app.cell(hide_code=True)
def text_exercise_rate():
    mo.md(r"""
    ### Oefening 2. De learning rate van fase 2
    """)
    return


@app.cell(hide_code=True)
def choice_rate_cell():
    choice_rate = L.choice(
        "big-lr",
        question="Een collega houdt de learning rate van fase 1 ook voor fase 2: 0,01 in plaats van 0,001. Wat verwacht je, over 5 seeds?",
        options={
            "faster": "Beter: met grotere stappen passen de geleende features zich sneller aan de spikkels aan",
            "right": "Slechter bij de meeste seeds: de features schuiven te ver op",
            "same": "Ongeveer hetzelfde: na fase 1 valt er weinig meer te leren",
        },
        correct="right",
        why={
            "faster": f"Sneller, maar te ver. De course mat het, over 5 seeds: {C.REFERENCE['big-lr'][0]} tot {C.REFERENCE['big-lr'][1]} met 0,01, tegen {C.REFERENCE['tuned'][0]} tot {C.REFERENCE['tuned'][1]} met 0,001, en {C.REFERENCE['frozen'][0]} tot {C.REFERENCE['frozen'][1]} voor fase 1 alleen.",
            "same": f"In fase 2 leert elke layer, en de learning rate bepaalt hoe ver elke stap gaat. De course mat het, over 5 seeds: {C.REFERENCE['big-lr'][0]} tot {C.REFERENCE['big-lr'][1]} met 0,01, tegen {C.REFERENCE['tuned'][0]} tot {C.REFERENCE['tuned'][1]} met 0,001.",
            "right": f"De course mat het, over 5 seeds: {C.REFERENCE['big-lr'][0]} tot {C.REFERENCE['big-lr'][1]} met 0,01, tegen {C.REFERENCE['tuned'][0]} tot {C.REFERENCE['tuned'][1]} met 0,001. Bij 4 van de 5 seeds doet 0,01 het zelfs slechter dan fase 1 alleen, {C.REFERENCE['frozen'][0]} tot {C.REFERENCE['frozen'][1]}: de geleende features verliezen een deel van wat 6.000 cijfers ze leerden.",
        },
    )
    choice_rate
    return (choice_rate,)


@app.cell(hide_code=True)
def text_exercise_more():
    mo.md(r"""
    ### Oefening 3. Tien keer zoveel voorbeelden
    """)
    return


@app.cell(hide_code=True)
def choice_more_cell():
    choice_more = L.choice(
        "more-digits",
        question=f"Met 100 gelabelde cijfers van de oude scanner: vanaf nul {C.REFERENCE['scratch'][0]} tot {C.REFERENCE['scratch'][1]} van de 1.000, fase 1 alleen {C.REFERENCE['frozen'][0]} tot {C.REFERENCE['frozen'][1]}, na fine-tuning {C.REFERENCE['tuned'][0]} tot {C.REFERENCE['tuned'][1]}. Nu krijg je 1.000 gelabelde cijfers. Wat verwacht je?",
        options={
            "scratch": "Vanaf nul wint nu: met 1.000 voorbeelden helpen geleende features niet meer",
            "only": "Alleen vanaf nul gaat erop vooruit: de andere kenden cijfers al",
            "right": "Alle drie gaan erop vooruit, en vanaf nul het meest: met 1.000 voorbeelden overfit het veel minder",
        },
        correct="right",
        why={
            "scratch": f"Niet met 1.000. De course mat het, over 5 seeds: vanaf nul {C.REFERENCE['1000-scratch'][0]} tot {C.REFERENCE['1000-scratch'][1]}, tegen {C.REFERENCE['1000-tuned'][0]} tot {C.REFERENCE['1000-tuned'][1]} na fine-tuning.",
            "only": f"Fase 1 alleen en fine-tuning gaan er ook op vooruit: {C.REFERENCE['1000-frozen'][0]} tot {C.REFERENCE['1000-frozen'][1]} en {C.REFERENCE['1000-tuned'][0]} tot {C.REFERENCE['1000-tuned'][1]}, tegen {C.REFERENCE['1000-scratch'][0]} tot {C.REFERENCE['1000-scratch'][1]} vanaf nul. De course mat het, over 5 seeds.",
            "right": f"De course mat het, over 5 seeds: vanaf nul {C.REFERENCE['1000-scratch'][0]} tot {C.REFERENCE['1000-scratch'][1]}, fase 1 alleen {C.REFERENCE['1000-frozen'][0]} tot {C.REFERENCE['1000-frozen'][1]}, na fine-tuning {C.REFERENCE['1000-tuned'][0]} tot {C.REFERENCE['1000-tuned'][1]}. Met tien keer zoveel voorbeelden overfitten 52.000 getallen veel minder; de geleende features helpen nog steeds, en fine-tuning voegt nog steeds een beetje toe.",
        },
    )
    choice_more
    return (choice_more,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Fase 1 is [Een pretrained netwerk als feature extractor](/course/open/torch-frozen-features?lang=nl); in fase 2 laat `requires_grad` uit [PyTorch-autograd doet wat jouw Value deed](/course/open/torch-autograd?lang=nl) elke layer weer leren, met een learning rate zoals in [Te klein, te groot: de learning rate](/course/open/ml-learning-rate?lang=nl), nu voor features die al goed zijn. Later in de course finetune je je eigen language model, getraind op veel verhalen, om instructies te volgen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_tune, choice_rate, choice_more):
    L.wrap_up(result_tune, choice_rate, choice_more)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
