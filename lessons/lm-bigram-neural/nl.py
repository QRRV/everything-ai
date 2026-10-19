# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les lm-bigram-neural. Route: course/route/lessen/taal.yaml. Checks: checks.py.
# termen:allow regularization

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("lm-bigram-neural", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [One-hot-vectoren voor klassen](/course/open/ml-one-hot?lang=nl) en [Laat de optimizer de stap zetten](/course/open/torch-optimizer?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_eye():
    quick_eye = L.choice(
        "quick-eye",
        question="Wat geeft `np.eye(4)[2]`?",
        options={
            "two": "`[0, 2, 0, 0]`: een 2 op plaats 2",
            "right": "`[0, 0, 1, 0]`: de one-hot-vector van klasse 2",
            "row": "`[1, 1, 1, 1]`: rij 2 van een matrix met enen",
        },
        correct="right",
        why={
            "two": "Een one-hot-vector bevat één enkele 1, niet het nummer van de klasse. Zie [One-hot-vectoren voor klassen](/course/open/ml-one-hot?lang=nl).",
            "row": "`np.eye(4)` is de eenheidsmatrix: alleen enen op de diagonaal. Zie [One-hot-vectoren voor klassen](/course/open/ml-one-hot?lang=nl).",
            "right": "Rij 2 van de eenheidsmatrix heeft haar enige 1 op plaats 2, geteld vanaf 0: de one-hot-vector van klasse 2.",
        },
    )
    quick_eye
    return (quick_eye,)


@app.cell(hide_code=True)
def quick_check_zero():
    quick_zero = L.choice(
        "quick-zero",
        question="Wat doet `opt.zero_grad()` in een training loop?",
        options={
            "weights": "Het zet de weights terug op 0, zodat elke stap met dezelfde tabel begint",
            "loss": "Het zet de loss terug op 0, zodat die niet van stap tot stap optelt",
            "right": "Het wist de hellingen, zodat `backward` niet bij de oude optelt",
        },
        correct="right",
        why={
            "weights": "De weights blijven zoals ze zijn; alleen hun hellingen worden gewist. Zie [Laat de optimizer de stap zetten](/course/open/torch-optimizer?lang=nl).",
            "loss": "De loss wordt toch elke stap opnieuw uitgerekend. `zero_grad` gaat over de hellingen. Zie [Laat de optimizer de stap zetten](/course/open/torch-optimizer?lang=nl).",
            "right": "`opt.zero_grad()` zet de `.grad` van elke parameter op `None`. `backward` telt zijn hellingen op bij wat er al staat, dus zonder `zero_grad` zou elke stap ook de hellingen van alle stappen ervoor meenemen.",
        },
    )
    quick_zero
    return (quick_zero,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Je bigram-model is een tabel die je hebt geteld. Tellen werkt voor paren, maar een model dat verder terugkijkt, heeft veel te veel combinaties om te tellen, en een neural network telt helemaal niet: het leert zijn getallen met gradient descent. In deze les leert een netwerk met één layer een bigram-tabel uit dezelfde paren, met gradient descent in plaats van tellen. Het is de eerste stap van tellen naar leren, en elk model in de rest van je project leert zo. Dit is de derde stap van je eigen LLM.
        """),
        draw.project_steps(["Letters en paren tellen", "Een bigram-model door te tellen", "Hetzelfde model, geleerd door een neural network",
                            "Je eigen tokenizer", "Je eigen GPT", "Een klein LLM pretrainen", "Het LLM instructies en voorkeuren laten volgen"],
                           2, here_label="← vandaag", title="Waar je project staat: je eigen LLM"),
        L.course_versions(),
        mo.md(r"""
        **Een voorbeeld uit het dagelijks leven.** Twee manieren om te weten hoe lang je onderweg bent naar je werk: elke rit opschrijven en het gemiddelde nemen, of beginnen met een schatting en die na elke rit een beetje bijstellen, tot hij bijna niet meer verandert.
        """),
    ])
    return


@app.cell(hide_code=True)
def predict_nll():
    guess_nll = L.guess(
        "net-nll",
        question=f"Het netwerk begint met een tabel vol nullen: elk teken even waarschijnlijk na elk teken, een NLL van 4,17. Het traint {C.STEPS} stappen gradient descent op de paren van de eerste {C.grouped(C.START, L.lang)} tekens van de toneelstukken. De tabel die op diezelfde tekens geteld is, haalt daar een NLL van {C.number(C.counted_nll(), 2, L.lang)}. Wat denk je dat het netwerk haalt?",
        options={
            "better": "Ongeveer 1,5: leren is beter dan tellen",
            "right": "Ongeveer 2,4: dicht bij de getelde tabel",
            "far": "Ongeveer 3,5: er nog ver vandaan",
        },
        reveal=f"**Ongeveer 2,4: {C.number(C.net_nll(), 2, L.lang)}.** Dicht bij de getelde tabel, maar er niet onder: op de paren waarvan het leert, voorspelt geen enkele tabel beter dan de getelde. Gradient descent vindt geen betere tabel; het vindt dezelfde tabel op een andere manier, zoals de schatting die je na elke rit bijstelt, ongeveer bij het gemiddelde uitkomt.",
    )
    guess_nll
    return (guess_nll,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Het netwerk heeft één tabel met weights, $W$, 65 bij 65: een rij voor elk teken, een kolom voor elk volgend teken. Om te voorspellen wat er na een teken komt, neemt het de one-hot-rij van dat teken, $X$, en vermenigvuldigt die met de tabel: `X @ W`. Een one-hot-rij heeft één enkele 1, dus `X @ W` is één rij van $W$: de rij van dat teken. Haar 65 getallen zijn logits; softmax maakt er de kansen op het volgende teken van, en cross-entropy, de gemiddelde NLL van de paren, is de loss. De tabel met kansen van het netwerk is dus softmax van $W$, rij voor rij.

        Het leert $W$ met gradient descent: begin met $W$ helemaal nul, zodat elk volgend teken even waarschijnlijk is, en zet kleine stapjes bergaf op de loss. Een rij kiezen met een one-hot-rij geeft hetzelfde als haar kiezen met haar nummer, dus in PyTorch kun je ook `W[x]` schrijven, voor alle paren tegelijk, zonder de one-hot-rijen te maken.

        Tel een straf voor grote weights bij de loss op, zoals in [Regularisatie en weight decay](/course/open/ml-regularization?lang=nl), maar met het gemiddelde van de gekwadrateerde weights in plaats van hun som: het gemiddelde is de som gedeeld door 4.225, het aantal weights, dus $\lambda$ hoeft niet piepklein te zijn. Grote weights kosten iets, dus de logits blijven dichter bij 0, en elke rij kansen blijft dichter bij gelijk verdeeld. Dat deed add-k smoothing uit [Paren die je nooit zag: smoothing](/course/open/lm-smoothing?lang=nl) met de tellingen: een grotere $\lambda$ geeft, net als een grotere k, vlakkere kansen.
        """),
        L.formula(r"[[loss:\mathcal{L}]] = \text{cross-entropy}(X [[param:W]], y) + [[hyper:\lambda]] \cdot \text{mean}([[param:W]]^2)"),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | `F.cross_entropy(logits, y)` | | softmax en cross-entropy in één stap, gemiddeld over alle paren; `y` bevat het nummer van het goede volgende teken van elk paar. `F` is `torch.nn.functional` van PyTorch, geïmporteerd met `import torch.nn.functional as F` | `F.cross_entropy(W[x], y)` |
        | `W[x]` | W van x | de rijen van `W` voor de tekens in `x`, één rij per paar: hetzelfde als `torch.eye(65)[x] @ W` | `W[x]` |
        | `torch.zeros(shape, requires_grad=True)` | | een tensor met nullen waarvan PyTorch de hellingen volgt, zoals `np.zeros` maar in PyTorch | `torch.zeros((65, 65), requires_grad=True)` |
        | `torch.softmax(W, dim=1)` | | softmax van elke rij van `W` | |
        """),
        L.watch_out("De loss komt niet onder de NLL van de getelde tabel: op de paren waarvan het leert, voorspelt geen enkele tabel beter dan de tellingen. En ook zonder de straf geeft softmax nooit precies 0: een paar dat nooit voorkwam, krijgt een heel negatieve logit, en een heel kleine kans."),
        mo.md(f"""
        De figuur: de loss terwijl het netwerk traint op de eerste {C.grouped(C.START, L.lang)} tekens van de toneelstukken, met learning rate {C.LR}, en de NLL van de tabel die op diezelfde tekens geteld is, als stippellijn.
        """),
        mo.Html(C.loss_curve({"title": "De tabel leren", "steps": "stappen gradient descent", "net": "het netwerk",
                              "counted": f"de getelde tabel, {C.number(C.counted_nll(), 2, L.lang)}", "fmt": lambda v: C.number(v, 1, L.lang),
                              "alt": f"Een dalende lijn, de loss van het netwerk over {C.STEPS} stappen: van 4,17 aan het begin daalt hij snel in de eerste stappen, en dan langzaam naar de stippellijn van de getelde tabel bij {C.number(C.counted_nll(), 2, L.lang)}, en eindigt bij {C.number(C.net_nll(), 2, L.lang)}."})),
        mo.md(f"""
        De loss daalt snel in de eerste stappen en daarna langzaam. Na {C.STEPS} stappen is hij {C.number(C.net_nll(), 2, L.lang)}, tegen {C.number(C.counted_nll(), 2, L.lang)} voor de getelde tabel. Bij vaak voorkomende paren zijn de twee tabellen het al eens: na "t" heeft "h" {C.number(C.net_prob('t', 'h'), 3, L.lang)} in het netwerk en {C.number(C.counted_prob('t', 'h'), 3, L.lang)} in de tellingen. Zeldzame tekens leren langzamer: hun rijen krijgen weinig paren, en dus kleine hellingen.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om de bigram-tabel te leren met gradient descent:

    1. Maak van de tekst paren van tekennummers: `x` het teken ervoor, `y` het teken erna.
    2. Begin met een tabel vol nullen waarvan PyTorch de hellingen volgt, `W = torch.zeros((65, 65), requires_grad=True)`, en een optimizer ervoor, `opt = torch.optim.SGD([W], lr=lr)`.
    3. Elke stap: de loss met de straf, `loss = F.cross_entropy(W[x], y) + lam * (W ** 2).mean()`; dan `opt.zero_grad()`, `loss.backward()` en `opt.step()`.
    4. De geleerde tabel met kansen is `torch.softmax(W, dim=1)`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De paren van "abab ba", met drie tekens: " " is 0, "a" is 1 en "b" is 2. 100 stappen met learning rate 5 en $\lambda$ = 0,01. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_learn():
    import torch
    import torch.nn.functional as F

    # 1. de paren van "abab ba": x het teken ervoor, y het teken erna
    _x = torch.tensor([1, 2, 1, 2, 0, 2])
    _y = torch.tensor([2, 1, 2, 0, 2, 1])
    # 2. een tabel vol nullen, en een optimizer ervoor
    _W = torch.zeros((3, 3), requires_grad=True)
    _opt = torch.optim.SGD([_W], lr=5)
    for _step in range(100):
        # 3. de loss met de straf, dan op nul, backward en de stap
        _loss = F.cross_entropy(_W[_x], _y) + 0.01 * (_W ** 2).mean()
        _opt.zero_grad()
        _loss.backward()
        _opt.step()
    print(_loss.item())
    # 4. de geleerde tabel met kansen
    with torch.no_grad():
        print(torch.softmax(_W, dim=1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De getelde tabel van "abab ba" heeft de rijen 0, 0, 1 voor " " en voor "a", en 1/3, 2/3, 0 voor "b". De geleerde tabel komt er dicht bij: ongeveer 0,96 en 0,98 waar de tellingen 1 hebben, en 0,33 en 0,66 voor "b". Waar de tellingen 0 hebben, heeft hij iets meer dan 0: 100 stappen drukken die logits maar een eindje omlaag, en de straf trekt ze terug naar 0. De eerste regel is de loss van de laatste stap, ongeveer 0,37: de cross-entropy, ongeveer 0,34, plus de straf. De getelde tabel haalt op deze paren een NLL van ongeveer 0,32.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_row():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Een one-hot-rij maal de tabel
    """)
    return


@app.cell(hide_code=True)
def choice_row_cell():
    choice_row = L.choice(
        "one-hot-row",
        question="`W` is een tabel van 65 × 65, en `X` de one-hot-rij van teken 3: een 1 op plaats 3 en overal anders 0. Wat is `X @ W`?",
        options={
            "col": "Kolom 3 van `W`: de weights naar teken 3",
            "right": "Rij 3 van `W`: de logits na teken 3",
            "num": "Het ene getal `W[3, 3]`",
            "times": "Heel `W`, maal 3",
        },
        correct="right",
        why={
            "col": "Een rij links van `@` combineert de rijen van `W`: hier alleen rij 3, maal 1. Een kolom zou uit `W @` een kolom komen.",
            "num": "`X` heeft 65 getallen, dus `X @ W` ook: een hele rij, niet één vakje.",
            "times": "De 1 op plaats 3 kiest een rij; hij vermenigvuldigt niets met 3.",
            "right": "Elke rij van `W` wordt vermenigvuldigd met het getal in `X` op haar plaats, en alleen plaats 3 bevat een 1: wat overblijft, is rij 3.",
        },
    )
    choice_row
    return (choice_row,)


@app.cell(hide_code=True)
def text_exercise_loss():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De loss van een tabel

        `loss_of(W, x, y)` geeft de loss van de tabel `W` op de paren `x` en `y`, met one-hot-rijen zoals in *Het idee*. De one-hot-rijen en de laatste regel staan er al. Vervang de `...` door de logits van elk paar. De check roept `loss_of(torch.tensor([[0.0, 1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.5, 1.0]]), torch.tensor([0, 1, 2, 0]), torch.tensor([1, 0, 2, 2]))` aan, en dat moet ongeveer `0.7557` geven; hij probeert ook andere tabellen. Is je functie goed, dan bewaart de check hem in je project, als `work/projects/llm/bigram_net.py`.
        """),
        L.task("Laat `loss_of` de logits van elk paar uitrekenen uit de one-hot-rijen `X` en de tabel `W`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_loss():
    box_loss = L.code_box("loss-of", lines=9)
    box_loss
    return (box_loss,)


@app.cell(hide_code=True)
def check_loss(box_loss):
    result_loss = L.check("loss-of", box_loss.value)
    result_loss
    return (result_loss,)


@app.cell(hide_code=True)
def hints_loss():
    L.hints(
        "loss-of",
        [
            "Het eerste deel van *Het idee*: wat geeft een one-hot-rij maal de tabel?",
            "`X` heeft één one-hot-rij per paar. Die vermenigvuldigen met `W` geeft één rij logits per paar.",
            "`logits = X @ ___`.",
        ],
        why="De rijen van `W` voor de tekens 0, 1, 2 en 0 zijn de logits van de vier paren. Cross-entropy geeft elk de verrassing van zijn goede volgende teken, 1, 0, 2 en 2, en neemt het gemiddelde: ongeveer 0,7557.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_train():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Train de tabel

        `train(x, y, steps, lr, lam)` leert de tabel uit de paren `x` en `y`, gegeven als lijsten met tekennummers. De tabel vol nullen, de optimizer en de loop staan er al. Vervang de `...` door stap 3 van het recept. De check traint op de paren van "the cat sat on the mat", 20 stappen met learning rate 10 en `lam` 0, en vergelijkt je tabel met die van de course via hun NLL op die paren, die ongeveer `0.6288` moet zijn; hij probeert ook andere aantallen stappen, learning rates en waarden van `lam`. Is je functie goed, dan bewaart de check hem in je project, als `work/projects/llm/bigram_train.py`.
        """),
        L.task("Laat `train` elke ronde van de loop één stap gradient descent zetten op de loss met de straf."),
    ])
    return


@app.cell(hide_code=True)
def exercise_train():
    box_train = L.code_box("train", lines=14)
    box_train
    return (box_train,)


@app.cell(hide_code=True)
def check_train(box_train):
    result_train = L.check("train", box_train.value)
    result_train
    return (result_train,)


@app.cell(hide_code=True)
def hints_train():
    L.hints(
        "train",
        [
            "Stap 3 van het recept, zoals in het uitgewerkte voorbeeld: vier regels in de loop.",
            "Eerst de loss, met de straf `lam` maal het gemiddelde van de gekwadrateerde weights. Dan de drie regels uit [Laat de optimizer de stap zetten](/course/open/torch-optimizer?lang=nl): de hellingen wissen, ze uitrekenen, de stap zetten.",
            "`loss = F.cross_entropy(W[x], y) + lam * (W ** 2).mean()`, dan `opt.zero_grad()`, `loss.backward()` en `opt.___()`.",
        ],
        why="Met `lam` 0 is de loss alleen de cross-entropy, en na 20 stappen met learning rate 10 voorspelt de tabel de paren van de korte tekst al goed: een NLL van ongeveer 0,63, tegen 4,17 voor de tabel vol nullen.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### Wat je project nu kan

    De cel hieronder traint een tabel met je `train`, op de paren van de eerste {C.grouped(C.START, L.lang)} tekens van de toneelstukken, {C.STEPS} stappen met learning rate {C.LR} en `lam` 0, en laat de geleerde tabel schrijven met je `generate` uit [Sampling: laat het model schrijven](/course/open/lm-sampling?lang=nl), of die van de course als de jouwe ontbreekt of stuk is. Hij werkt zodra oefening 2 en 3 af zijn, en het trainen duurt een paar seconden.
    """)
    return


@app.cell(hide_code=True)
def demo_learn(result_loss, result_train):
    result_loss, result_train  # run again when exercise 2 or 3 saves your module
    _result = C.demo()
    if _result is None:
        _out = mo.md("*Maak eerst oefening 2 en 3 af: dan staan je `loss_of` en je `train` in je project, en gebruikt deze cel ze.*")
    elif isinstance(_result, str):
        _out = mo.md({
            "slow": "*Trainen met je `train` duurt langer dan 30 seconden. Gaat elke stap één keer over de paren, met `W[x]`?*",
            "train": "*De `train` in `work/projects/llm/bigram_train.py` geeft niet meer wat de check verwacht, of stopt met een fout. Heb je dat bestand veranderd? Verwijder het, en laat oefening 3 opnieuw nakijken: die check bewaart je functie dan opnieuw.*",
            "net": "*De `loss_of` in `work/projects/llm/bigram_net.py` geeft niet meer wat de check verwacht, of stopt met een fout. Heb je dat bestand veranderd? Verwijder het, en laat oefening 2 opnieuw nakijken: die check bewaart je functie dan opnieuw.*",
            "sampling": "*Schrijven met `generate` uit `work/projects/llm/sampling.py` stopt met een fout. Heb je dat bestand veranderd? Verwijder het, en laat oefening 3 van [Sampling: laat het model schrijven](/course/open/lm-sampling?lang=nl) opnieuw nakijken.*",
        }[_result])
    else:
        _who = "" if _result["yours"] else " De tekst komt van de `generate` van de course: de jouwe ontbreekt of is stuk, zoals de melding bij *Waar je project staat* zegt."
        _out = mo.vstack([
            mo.md(f"""Je geleerde tabel haalt een NLL van {C.number(_result['nll'], 2, L.lang)} op de paren waarvan hij leerde, uitgerekend met je `loss_of`, tegen {C.number(C.counted_nll(), 2, L.lang)} voor de getelde tabel. Na "t" heeft "h" {C.number(_result['th'], 3, L.lang)}. Wat hij schrijft, beginnend met "T", met seed 1:"""),
            mo.Html(C.text_block(_result["text"])),
            mo.md(f"""Hij schrijft zoals je getelde model: het is dezelfde soort tabel, alleen geleerd in plaats van geteld, en uit een kleiner deel van de toneelstukken.{_who}"""),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De one-hot-rijen uit [One-hot-vectoren voor klassen](/course/open/ml-one-hot?lang=nl), de cross-entropy uit [Cross-entropy-loss voor veel klassen](/course/open/ml-cross-entropy?lang=nl) en de optimizer uit [Laat de optimizer de stap zetten](/course/open/torch-optimizer?lang=nl) leren samen de tabel die je eerder telde, en de straf uit [Regularisatie en weight decay](/course/open/ml-regularization?lang=nl) doet wat smoothing deed. Je project `llm` heeft nu ook `bigram_net.py` en `bigram_train.py`. In de volgende les, *Base model of chat model?*, zie je wat de language models die je elke dag gebruikt, anders doen dan het model dat jij bouwde.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_row, result_loss, result_train):
    L.wrap_up(choice_row, result_loss, result_train)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
