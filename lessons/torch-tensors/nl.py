# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les torch-tensors. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("torch-tensors", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Je MLP trainen: de training loop](/course/open/nn-micrograd-train?lang=nl) en [Broadcasting](/course/open/np-broadcasting?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_backward():
    quick_backward = L.choice(
        "quick-backward",
        question="Je `train` uit [Je MLP trainen: de training loop](/course/open/nn-micrograd-train?lang=nl) traint `MLP(1, [8, 1])` op 12 voorbeelden. Hoe vaak roept één stap van de loop `backward` aan?",
        options={
            "examples": "12: één keer per voorbeeld",
            "right": "1: één keer, op de loss van alle 12 voorbeelden",
            "params": "25: één keer per parameter, zodat elke parameter zijn eigen helling krijgt",
        },
        correct="right",
        why={
            "examples": "`mse` zet alle 12 voorbeelden eerst in één loss. Zie het recept in [Je MLP trainen: de training loop](/course/open/nn-micrograd-train?lang=nl).",
            "params": "Eén aanroep geeft elke parameter tegelijk zijn helling. Zie het recept in [Je MLP trainen: de training loop](/course/open/nn-micrograd-train?lang=nl).",
            "right": "`mse` telt alle 12 voorbeelden op in één loss, en één `backward(loss)` geeft alle 25 parameters hun helling.",
        },
    )
    quick_backward
    return (quick_backward,)


@app.cell(hide_code=True)
def quick_check_rows():
    quick_rows = L.choice(
        "quick-rows",
        question="`m` heeft shape (2, 3), en `t = m.sum(axis=1)`. Waarom laat `m / t` NumPy stoppen?",
        options={
            "right": "`t` heeft shape (2,), en rechts uitgelijnd verschillen 3 en 2",
            "sum": "NumPy kan niet door een som delen",
            "columns": "`t` heeft shape (3,): één totaal per kolom",
        },
        correct="right",
        why={
            "sum": "NumPy deelt door elke array waarvan de shape past. Wat is de shape van `t`? Zie de regel in [Broadcasting](/course/open/np-broadcasting?lang=nl).",
            "columns": "`axis=1` loopt langs elke rij, dus er is één totaal per rij. Zie [Broadcasting](/course/open/np-broadcasting?lang=nl).",
            "right": "Eén totaal per rij geeft shape (2,), en rechts uitgelijnd zetten (2, 3) en (2,) 3 tegenover 2. `keepdims=True` houdt `t` als kolom, (2, 1), en dan past het.",
        },
    )
    quick_rows
    return (quick_rows,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je project in [Je MLP trainen: de training loop](/course/open/nn-micrograd-train?lang=nl) traint een MLP met 25 parameters, en elk getal erin is een eigen `Value`: elk product is een Python-object met een lijst van zijn ouders. Een klein netwerk dat handgeschreven cijfers leest, heeft ongeveer 100.000 parameters en leert van 60.000 afbeeldingen. Met één `Value` tegelijk duurt één stap over al die afbeeldingen uren. NumPy rekent met hele arrays tegelijk, zoals in [Broadcasting](/course/open/np-broadcasting?lang=nl), maar het kan je geen hellingen geven, en het draait alleen op de CPU, de hoofdprocessor van de computer. PyTorch kan allebei. Zijn arrays heten tensors: ze rekenen zoals NumPy, ze kunnen op een GPU staan, een graphics processing unit die duizenden berekeningen tegelijk doet, en vanaf de volgende les onthouden ze waar ze vandaan komen, zoals je `Value` deed.

    **Een voorbeeld uit het dagelijks leven.** Een vaatwasser wast een hele lading tegelijk, veel sneller dan elk bord apart met de hand. Maar hij wast alleen wat erin staat: een bord dat nog op tafel staat, blijft vuil.
    """)
    return


@app.cell(hide_code=True)
def predict_dtype():
    guess_dtype = L.guess(
        "dtype",
        question="`np.array([0.1, 0.2])` heeft dtype float64: elk getal neemt 64 bits in. Welk dtype denk je dat `torch.tensor([0.1, 0.2])` heeft?",
        options={
            "same": "float64, hetzelfde als NumPy",
            "right": "float32: 32 bits per getal",
            "python": "Geen dtype: een tensor houdt de eigen floats van Python",
        },
        reveal="**float32.** PyTorch bewaart een float in 32 bits, de helft van de 64 van NumPy. Dat zijn ongeveer 7 cijfers in plaats van ongeveer 16: nauwkeurig genoeg om een netwerk te trainen, in de helft van het geheugen, en een GPU rekent er veel sneller mee. Daarom geeft `torch.tensor(0.1).item()` `0.10000000149011612`: de float32 die het dichtst bij 0,1 ligt, geschreven als Python-float. Gehele getallen zijn in allebei int64.",
    )
    guess_dtype
    return (guess_dtype,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        PyTorch is een library voor neural networks, die je importeert met `import torch`. Een **tensor** is een rooster van getallen met een willekeurig aantal axes: één getal heeft 0 axes, een vector 1, een matrix 2, en een batch kleurenfoto's 4 (foto, kleur, rij, kolom). Het is de array van PyTorch. Je maakt er een van een lijst met `torch.tensor(...)`, zoals `np.array(...)` een array maakt, en zijn `.shape`, `.dtype` en `.T` werken zoals je ze kent van NumPy.

        Een tensor rekent zoals een array: `+`, `-`, `*` en `/` getal voor getal, `@` voor matrixvermenigvuldiging, en broadcasting met dezelfde regel. Samenvattingen zoals `.sum()` en `.mean()` zijn methods van de tensor. Alleen de naam van een axis verschilt: PyTorch noemt hem een `dim`. `t.mean(dim=1)` neemt het gemiddelde langs elke rij, en `keepdim=True` houdt die axis met lengte 1, zoals `keepdims=True` in NumPy. PyTorch begrijpt ook `axis=` en `keepdims=` van NumPy, maar zijn eigen documentatie, en de meeste code die je zult lezen, gebruikt `dim` en `keepdim`. De mean squared error van een lineair model heeft daarom dezelfde formule en dezelfde regels code als met NumPy:
        """),
        L.formula(r"[[pred:\hat{\mathbf{y}}]] = \mathbf{X}[[param:\mathbf{w}]] + [[param:b]], \qquad [[loss:\mathcal{L}]] = \frac{1}{n} \sum_{i=1}^{n} \left(y_i - [[pred:\hat{y}_i]]\right)^2"),
        mo.md(r"""
        In code: `y_pred = X @ w + b` en `loss = ((y - y_pred) ** 2).mean()`.

        **Float32 en int64.** Een float die je in een tensor zet, wordt standaard **float32**: 32 bits, ongeveer 7 cijfers. Een tensor gemaakt van een NumPy-array houdt de float64 van NumPy. Een geheel getal is int64. PyTorch neemt alleen een gemiddelde van floats, dus het gemiddelde van een tensor met gehele getallen laat hem stoppen; `t.float()` geeft een kopie in float32. Ook in `@` mengt hij de twee niet. Geef data dus meteen het dtype dat ze nodig hebben: `torch.tensor(data, dtype=torch.float32)`.

        **Een tensor zonder axes.** `t.sum()` en `t.mean()` geven een tensor terug, geen getal: `tensor(0.4375)`. Dat is met opzet: in de volgende les onthoudt zo'n tensor hoe hij gemaakt is. `.item()` haalt het getal eruit, als Python-getal, voor een `print` of een lijst met losses.

        **Waar een tensor staat.** Elke tensor staat op een **device**: de CPU, `"cpu"`, of een GPU: `"cuda"` op een kaart van NVIDIA, `"mps"` op een Mac met een chip van Apple. `t.device` zegt op welk device hij staat. Een GPU doet duizenden berekeningen tegelijk, dus op een grote tensor is hij vele malen sneller. Eén regel: de tensors van één berekening moeten op hetzelfde device staan. PyTorch verplaatst een tensor nooit uit zichzelf; hij stopt met een fout zoals "Expected all tensors to be on the same device". Tot *Waarom AI GPU's gebruikt, en dezelfde code op de jouwe*, een paar lessen verderop, blijven je tensors op de CPU; die les laat zien hoe je ze verplaatst.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **tensor** | | een rooster van getallen met een willekeurig aantal axes: 0 voor één getal, 1 voor een vector, 2 voor een matrix. De array van PyTorch | `torch.tensor([[1, 2], [3, 4]])` |
        | **float32** | float tweeëndertig | een float in 32 bits, ongeveer 7 cijfers: de standaard van PyTorch voor floats, de helft van het geheugen van float64 in NumPy | `dtype=torch.float32` |
        | `dim` | | de naam van PyTorch voor een axis; `keepdim=True` houdt hem met lengte 1 | `t.mean(dim=1, keepdim=True)` |
        | `.item()` | | het getal in een tensor zonder axes, als Python-getal | `loss.item()` |
        | **device** | | de plek waar een tensor staat en waarop ermee gerekend wordt: de CPU, of een GPU | `t.device` |
        """),
        L.watch_out("In de natuurkunde en de zuivere wiskunde is een tensor iets strengers: een object dat op een vaste manier verandert als je de coördinaten verandert. In machine learning is een tensor gewoon een rooster van getallen met een willekeurig aantal axes. Kom je het woord tegen in een tekst over natuurkunde, dan betekent het daar meer."),
        mo.md(r"""
        Drie tensors uit het uitgewerkte voorbeeld hieronder, met 0, 1 en 2 axes. PyTorch schrijft een shape als `torch.Size`.
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_tensors():
    draw.array_grids(C.picture(["0 axes", "1 axis", "2 axes"]),
                     title="Tensors met 0, 1 en 2 axes", box_width=66)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om met PyTorch te rekenen in plaats van met NumPy:

    1. Maak van de data tensors, met floats: `torch.tensor(data, dtype=torch.float32)`.
    2. Reken zoals met NumPy: `@`, broadcasting, en methods zoals `.mean(dim=..., keepdim=True)`.
    3. Voor één getal, zoals een loss, haal je het eruit met `.item()`.
    4. Kijk met `.device` waar je tensors staan: de tensors van één berekening moeten allemaal op hetzelfde device staan.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een lineair model op 4 voorbeelden met elk 2 inputs: zijn voorspellingen en zijn mean squared error, met tensors. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_tensors():
    import torch

    # 1. de data als tensors van floats
    _X = torch.tensor([[1, 2], [3, 1], [0, 2], [2, 2]], dtype=torch.float32)   # (4, 2)
    _y = torch.tensor([5, 4, 3, 6], dtype=torch.float32)                       # (4,)
    _w = torch.tensor([1.0, 1.5])                                              # (2,)
    _b = 0.5
    # 2. reken zoals met NumPy: (4, 2) @ (2,) + een getal -> (4,)
    _y_pred = _X @ _w + _b
    _loss = ((_y - _y_pred) ** 2).mean()
    print(_y_pred)
    print(_loss)
    print(_X.mean(dim=0))   # het gemiddelde van elke kolom: één per input
    print(_X.mean(dim=1, keepdim=True))   # het gemiddelde van elke rij, als kolom gehouden: (4, 1)
    # 3. de loss als Python-getal
    print(_loss.item())
    # 4. waar de tensors staan, en of PyTorch een GPU van NVIDIA vindt
    print(_X.device, _y_pred.device)
    print(torch.cuda.is_available())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `y_pred` is 4,5; 5; 3,5 en 5,5. De residuen $y - \hat{y}$ zijn 0,5; −1; −0,5 en 0,5, hun kwadraten 0,25; 1; 0,25 en 0,25, en hun gemiddelde is 1,75 / 4 = 0,4375. `print(_loss)` laat `tensor(0.4375)` zien: nog steeds een tensor, zonder axes. `.item()` geeft het Python-getal `0.4375`. `_X.mean(dim=0)` loopt langs de rijen omlaag en geeft één gemiddelde per kolom: 1,5 en 1,75. `_X.mean(dim=1, keepdim=True)` loopt langs elke rij en geeft één gemiddelde per rij, 1,5; 2; 1 en 2, als kolom met shape (4, 1), zodat het voor broadcasting bij `_X` past. Beide tensors staan op de CPU, en de laatste regel zegt of PyTorch op je computer een GPU van NVIDIA vindt. Hoe dan ook draaien de volgende lessen op de CPU.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_device():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Twee devices
    """)
    return


@app.cell(hide_code=True)
def choice_device_cell():
    choice_device = L.choice(
        "same-device",
        question="Op een computer met een GPU van NVIDIA staat `X` op de GPU, `device(type='cuda', index=0)`, en `w` op de CPU. Wat doet `X @ w`?",
        options={
            "moves": "PyTorch zet `w` uit zichzelf op de GPU, omdat `X` daar al staat, en rekent op de GPU",
            "cpu": "PyTorch rekent op de CPU, waar `w` staat",
            "right": "PyTorch stopt met een fout: de twee tensors staan op verschillende devices",
        },
        correct="right",
        why={
            "moves": "Een tensor verplaatsen kost tijd, dus PyTorch laat aan jou over waar hij heen gaat. Zie *Waar een tensor staat* in *Het idee*.",
            "cpu": "Dan zou hij `X` moeten verplaatsen. Verplaatst PyTorch een tensor uit zichzelf? Zie *Waar een tensor staat* in *Het idee*.",
            "right": "\"Expected all tensors to be on the same device\": de tensors van één berekening moeten op hetzelfde device staan, en PyTorch verplaatst er nooit een uit zichzelf.",
        },
    )
    choice_device
    return (choice_device,)


@app.cell(hide_code=True)
def text_exercise_loss():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De loss als getal

        `X` bevat 3 voorbeelden met elk 2 inputs, shape (3, 2), `y` hun labels, shape (3,), en `w` en `b` zijn de weights en de bias term van een lineair model. Vervang de twee `...`: `y_pred` de voorspellingen voor alle voorbeelden tegelijk, en `loss` hun mean squared error, als Python-getal. De check probeert ook andere weights en bias terms.
        """),
        L.task("Maak `loss` de mean squared error van `X @ w + b`, als Python-getal."),
    ])
    return


@app.cell(hide_code=True)
def exercise_loss():
    box_loss = L.code_box("loss", lines=11)
    box_loss
    return (box_loss,)


@app.cell(hide_code=True)
def check_loss(box_loss):
    result_loss = L.check("loss", box_loss.value)
    result_loss
    return (result_loss,)


@app.cell(hide_code=True)
def hints_loss():
    L.hints(
        "loss",
        [
            "Stap 2 en 3 van het recept, zoals in het uitgewerkte voorbeeld.",
            "De voorspellingen zijn een matrix keer een vector, plus een getal. De mean squared error blijft een tensor zonder axes tot je het getal eruit haalt.",
            "`y_pred = X @ w + b` en `loss = ((y - y_pred) ** 2).___().___()`.",
        ],
        why="$\\hat{\\mathbf{y}} = (2 + 2 - 1;\\ 0 + 6 - 1;\\ 1 + 2 - 1) = (3; 5; 2)$. De residuen $y - \\hat{y}$ zijn 1, −2 en 1, dus het gemiddelde van hun kwadraten is (1 + 4 + 1) / 3 = 2.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_centered():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Elke rij min zijn eigen gemiddelde

        `scores` bevat de cijfers van twee leerlingen voor drie toetsen, als gehele getallen, met een rij per leerling: shape (2, 3). Vervang de `...`: `centered` is `scores` met elke rij min het eigen gemiddelde van die rij, zodat je per leerling ziet welke toets beter of slechter ging dan gewoonlijk. De check probeert ook andere scores.
        """),
        L.task("Laat `centered` elke score min het gemiddelde van zijn eigen rij bevatten."),
    ])
    return


@app.cell(hide_code=True)
def exercise_centered():
    box_centered = L.code_box("centered", lines=6)
    box_centered
    return (box_centered,)


@app.cell(hide_code=True)
def check_centered(box_centered):
    result_centered = L.check("centered", box_centered.value)
    result_centered
    return (result_centered,)


@app.cell(hide_code=True)
def hints_centered():
    L.hints(
        "centered",
        [
            "Stap 1 en 2 van het recept: gehele getallen worden eerst floats, en dan neem je één gemiddelde per rij.",
            "Eén gemiddelde per rij is een samenvatting langs elke rij. Om het van een matrix af te trekken, houd je het als kolom, zoals `keepdims` deed in [Broadcasting](/course/open/np-broadcasting?lang=nl).",
            "`centered = scores - scores.float().mean(dim=___, keepdim=___)`.",
        ],
        why="De rijen hebben de gemiddelden 7 en 6. (6; 8; 7) − 7 = (−1; 1; 0), en (4; 9; 5) − 6 = (−2; 3; −1).",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Een tensor is de array die je kent uit [Broadcasting](/course/open/np-broadcasting?lang=nl): dezelfde shapes, dezelfde `@` en dezelfde regel voor het uitrekken. Wat nieuw is, is klein: float32, `dim`, `.item()` en het device. In [Je MLP trainen: de training loop](/course/open/nn-micrograd-train?lang=nl) was elk getal een eigen `Value`; één tensor bevat duizenden getallen in één object. In de volgende les, *PyTorch-autograd doet wat jouw Value deed*, leert een tensor te onthouden waar hij vandaan komt, en geeft `backward()` elk getal erin zijn helling, zoals je eigen `Value` deed.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_device, result_loss, result_centered):
    L.wrap_up(choice_device, result_loss, result_centered)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
