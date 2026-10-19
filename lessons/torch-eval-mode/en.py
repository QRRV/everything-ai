# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson torch-eval-mode. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("torch-eval-mode", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Datasets and DataLoaders](/course/open/torch-dataloader?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_steps():
    quick_steps = L.choice(
        "quick-steps",
        question="`loader = DataLoader(ds, batch_size=50, shuffle=True)`, and `ds` holds 200 examples. How many update steps do 3 epochs take, with a step for every mini-batch?",
        options={
            "epochs": "3: one per epoch",
            "examples": "600: one per example, every epoch",
            "right": "12: one per mini-batch, every epoch",
        },
        correct="right",
        why={
            "epochs": "Each epoch goes over all mini-batches of the DataLoader, with a step for each. See [Datasets and DataLoaders](/course/open/torch-dataloader?lang=en).",
            "examples": "A step takes a whole mini-batch of 50 at once. See [Datasets and DataLoaders](/course/open/torch-dataloader?lang=en).",
            "right": "200 / 50 = 4 mini-batches per epoch, so 3 epochs take 4 · 3 = 12 steps.",
        },
    )
    quick_steps
    return (quick_steps,)


@app.cell(hide_code=True)
def quick_check_item():
    quick_item = L.choice(
        "quick-item",
        question="`ds` is a Dataset, as in [Datasets and DataLoaders](/course/open/torch-dataloader?lang=en). What does `ds[3]` give?",
        options={
            "right": "Example 3: its input and its label",
            "batch": "The mini-batch with number 3",
            "first": "The first 3 examples",
        },
        correct="right",
        why={
            "batch": "Mini-batches come from the DataLoader, not from the Dataset. See *The idea* in [Datasets and DataLoaders](/course/open/torch-dataloader?lang=en).",
            "first": "`ds[3]` runs `__getitem__` with `i` = 3. See *The idea* in [Datasets and DataLoaders](/course/open/torch-dataloader?lang=en).",
            "right": "`ds[3]` runs `ds.__getitem__(3)`, which gives back the input and the label of example 3.",
        },
    )
    quick_item
    return (quick_item,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    So far, the networks in this course were mostly being trained: a loss, a backward pass and a step. Once a model is trained, you use it: it predicts for new inputs, and nothing in it changes. Two things are different then. Some layers do something on purpose while the network learns, and must stop doing it when it predicts. And PyTorch records a graph for every calculation with a parameter, which costs memory and time, while a prediction needs no slopes. PyTorch has a switch for each.

    **An everyday example.** In a rehearsal, the director makes things harder on purpose: a few actors sit out a scene, so the others learn to play it without them, and an assistant writes down every slip for the notes afterwards. On the night of the show everyone plays, and nobody writes anything down: the actors just perform what they learned.
    """)
    return


@app.cell(hide_code=True)
def predict_three():
    guess_three = L.guess(
        "three",
        question="A trained network has a layer called dropout, which *The idea* below describes. It is still in the mode it trained in. You ask it three times for its prediction at $x = 1$. What do you think you get?",
        options={
            "same": "Three times the same number",
            "right": "Three different numbers",
            "error": "An error: a network that is still training cannot predict",
        },
        reveal="**Three different numbers.** In the mode it trains in, dropout sets each number of the hidden layer to 0 by chance, a new chance at every call. So the same input gives another output each time. In the worked example below, the three calls give 2.62, 2.33 and 2.04. Switched to the mode for predicting, dropout passes everything through, and every call gives 2.56.",
    )
    guess_three
    return (guess_three,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Using a trained model to predict for new inputs is **inference**, as in [Examples, features and labels](/course/open/ml-vocabulary?lang=en). For it, a network needs two switches, and they do different things.

        **Train mode and eval mode.** Every module, from one `nn.Linear` to a class built on `nn.Module`, has a switch: `model.training`. It starts at `True`: **train mode**. `model.eval()` sets it to `False`, **eval mode**, for the model and every layer in it, and `model.train()` sets it back. Most layers ignore the switch: `nn.Linear` and ReLU calculate the same in both modes. A few layers read it. **Dropout**, `nn.Dropout(0.5)`, is one: in train mode it sets each number that passes through it to 0 with a chance of 0.5, and doubles the others; in eval mode it passes everything through as it is. A later lesson explains why such randomness helps a network learn. For a prediction it only gets in the way.

        **No graph: `torch.no_grad()`.** As in [PyTorch autograd does what your Value did](/course/open/torch-autograd?lang=en), PyTorch records nothing inside `with torch.no_grad():`; there it kept the update step out of the graph, and now `opt.step()` does that for you. A prediction needs no slopes either, so its forward pass can skip the graph too, and save the memory and time that recording takes. The outputs then say `requires_grad=False`.

        The two switches are independent: `model.eval()` does not stop the recording, and `torch.no_grad()` does not switch off dropout. A prediction uses both. And a network that predicts in between training steps has to go back to train mode before the next step, with `model.train()`.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **train mode** | | the mode a network trains in: layers such as dropout do their work | `model.train()` |
        | **eval mode** | | the mode a network predicts in: layers such as dropout pass everything through | `model.eval()` |
        | `model.training` | | the switch itself: `True` in train mode, `False` in eval mode | `net.training` |
        | `nn.Dropout(p)` | n n dropout | a layer that, in train mode, sets each number to 0 with chance `p`; explained in a later lesson | `nn.Dropout(0.5)` |
        """),
        L.watch_out("In statistics, inference means something else: statistical inference draws conclusions about a whole group from a sample of it. In machine learning, inference is simply using a trained model to predict. If you read the word in a statistics text, it usually means statistical inference."),
        mo.md(r"""
        The picture: the network of the worked example, with dropout, trained on the 200 examples of [Mini-batches and epochs](/course/open/torch-batches?lang=en). Its predictions for $x$ from −2 to 2: three times in train mode, and once in eval mode.
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_modes():
    _train_loss, _eval_loss = C.trained()[3], C.trained()[4]
    mo.vstack([
        mo.Html(C.picture({"title": "One network, two modes", "dashed": "dashed: train mode, three calls", "solid": "solid: eval mode",
                           "alt": f"200 points around the line y = 2x + 1. Three dashed lines, the network's predictions in train mode, zigzag differently around the points; one solid line, its prediction in eval mode, runs smoothly through them. The loss over all points is about {L.number(_train_loss, 1)} in train mode, a little different at every call, and {L.number(_eval_loss, 2)} in eval mode."})),
        mo.md(f"""
        In train mode every call gives a different line, and the loss over all points is about <span class="c-loss ea-num">{L.number(_train_loss, 1)}</span>, a little different at every call. In eval mode the line is the same at every call, and the loss is about <span class="c-loss ea-num">{L.number(_eval_loss, 2)}</span>. The network is the same; only the switch is set differently.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To predict with a network, also in between training steps:

    1. Switch to eval mode: `model.eval()`.
    2. Work out the predictions inside `with torch.no_grad():`.
    3. Switch back to train mode, `model.train()`, before any further training.
    4. Put steps 1 to 3 in a function `predict(model, X)`, so that you never forget one of them.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The network of the picture: 1 input, a hidden layer of 16 neurons with ReLU, dropout, and 1 output, trained for 15 epochs with mini-batches of 20. Then three calls in train mode, each predicting at $x = 0$ and $x = 1$, and two calls with the recipe. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_eval():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(1, 16)
            self.drop = nn.Dropout(0.5)
            self.out = nn.Linear(16, 1)

        def forward(self, x):
            return self.out(self.drop(torch.relu(self.hidden(x))))

    _X, _y = C.data()   # the 200 examples of Mini-batches and epochs
    torch.manual_seed(1)
    _net = Net()
    _opt = torch.optim.SGD(_net.parameters(), lr=0.05)
    _loader = DataLoader(TensorDataset(_X, _y), batch_size=20, shuffle=True)
    for _epoch in range(15):
        for _xb, _yb in _loader:
            _loss = ((_net(_xb) - _yb) ** 2).mean()
            _opt.zero_grad()
            _loss.backward()
            _opt.step()

    _points = torch.tensor([[0.0], [1.0]])   # x = 0 and x = 1
    print(_net.training)
    for _k in range(3):
        _out = _net(_points)   # still train mode: another answer every time
        print(_out[0].item(), _out[1].item(), _out.requires_grad)

    # 4. the recipe as a function
    def _predict(model, X):
        model.eval()               # 1. eval mode
        with torch.no_grad():      # 2. no graph
            out = model(X)
        model.train()              # 3. back to train mode
        return out

    for _k in range(2):
        _out = _predict(_net, _points)
        print(_out[0].item(), _out[1].item(), _out.requires_grad)
    print(_net.training)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    After training, `_net.training` is `True`: train mode, as every network starts. The three calls in train mode give three different pairs, about (0.99, 2.62), (1.02, 2.33) and (0.77, 2.04), and `requires_grad` is `True`: PyTorch recorded a graph for each. `_predict` gives about (0.87, 2.56) twice, the same both times, and `requires_grad` is `False`. After `_predict`, `_net.training` is `True` again. The predictions lie a little under the line $y = 2x + 1$, which gives 1 and 3. That is not for lack of epochs: a network that learns with dropout ends up with a line a little flatter than the data, as the solid line in the picture shows.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_switches():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Two switches
    """)
    return


@app.cell(hide_code=True)
def choice_switches_cell():
    choice_switches = L.choice(
        "two-switches",
        question="What does `model.eval()` do, and what does `torch.no_grad()` do?",
        options={
            "swapped": "`model.eval()` stops the recording; `torch.no_grad()` switches dropout off",
            "right": "`model.eval()` switches dropout off; `torch.no_grad()` stops the recording",
            "same": "Both do the same: they switch a network from learning to predicting",
        },
        correct="right",
        why={
            "same": "They are two different switches, and a prediction needs both. See *The idea*.",
            "swapped": "It is the other way round. See *Train mode and eval mode* in *The idea*.",
            "right": "`model.eval()` sets `model.training` to `False`, which dropout reads; `torch.no_grad()` keeps PyTorch from recording a graph. Neither does the other's job.",
        },
    )
    choice_switches
    return (choice_switches,)


@app.cell(hide_code=True)
def text_exercise_evaluate():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Measure the loss in eval mode

        To see how well a network does in between training steps, you measure its loss as it predicts. The class `Net` of the worked example is there. Finish `evaluate(model, X, y)`: steps 1 to 3 of the recipe, with the mean squared error of the predictions for `X` in step 2, and give back that loss. Replace the `...`. The last lines make a network and call `evaluate` on three points of the line $y = 2x + 1$, then show whether a graph was recorded, whether the network is in train mode again, and the loss.
        """),
        L.task("Make `evaluate` give back the loss of the network in eval mode, worked out without a graph, and switch the network back to train mode."),
    ])
    return


@app.cell(hide_code=True)
def exercise_evaluate():
    box_evaluate = L.code_box("evaluate", lines=24)
    box_evaluate
    return (box_evaluate,)


@app.cell(hide_code=True)
def check_evaluate(box_evaluate):
    result_evaluate = L.check("evaluate", box_evaluate.value)
    result_evaluate
    return (result_evaluate,)


@app.cell(hide_code=True)
def hints_evaluate():
    L.hints(
        "evaluate",
        [
            "Steps 1 to 3 of the recipe, as in `_predict` in the worked example; only what you work out inside the `with` block is different.",
            "Inside the `with` block: the mean of the squared differences between `model(X)` and `y`. Keep it in a name, switch back, and only then give it back: a line after `return` never runs.",
            "`model.___()`, then `with torch.no_grad():` with `loss = ((model(X) - y) ** 2).mean()`, then `model.___()` and `return loss`.",
        ],
        why="`model.eval()` switches dropout off, so the loss is that of the network as it predicts; `torch.no_grad()` records no graph; and `model.train()` leaves the network ready to train on. So the last line shows `False`, `True` and the loss in eval mode.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_when():
    mo.md(r"""
    ### Exercise 3. Training or predicting?

    Six lines from training and predicting code. What is each line for: only for training, only for predicting, or for both?
    """)
    return


@app.cell(hide_code=True)
def exercise_when():
    sort_when = L.sort("when")
    sort_when
    return (sort_when,)


@app.cell(hide_code=True)
def check_when(sort_when):
    result_when = L.check("when", sort_when)
    result_when
    return (result_when,)


@app.cell(hide_code=True)
def hints_when():
    L.hints(
        "when",
        [
            "For each line, ask: is it needed to learn, to make a prediction stable and cheap, or for every use of the network?",
            "Slopes are only for learning. And each switch of *The idea* has a position for training and one for predicting.",
            "`opt.zero_grad()` and `loss.backward()` are for training; `model.eval()` and `torch.no_grad()` for predicting. Which side is `model.train()` on, and which line do both need?",
        ],
        why="**Only for training:** `model.train()`, `opt.zero_grad()` and `loss.backward()`. **Only for predicting:** `model.eval()`, and `with torch.no_grad():` around the forward pass. **For both:** the forward pass `y_pred = model(X)`. `model.train()` also stands at the end of `evaluate`, but there it is for the training that comes after.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The training loop of [Datasets and DataLoaders](/course/open/torch-dataloader?lang=en) now has a counterpart for using the network: eval mode for layers such as dropout, and `torch.no_grad()` from [PyTorch autograd does what your Value did](/course/open/torch-autograd?lang=en) for the graph. In the next lesson, *Project: an MLP that reads handwritten digits*, you put everything together: a module, an optimizer, a DataLoader, and predictions in eval mode to see how many digits it reads right.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_switches, result_evaluate, result_when):
    L.wrap_up(choice_switches, result_evaluate, result_when)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
