# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-federated-learning. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-federated-learning", "en")
    C = L.checks   # ten phones with QMNIST digits, DigitsMLP(64), and FedAvg with the learner's fedavg_states


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Three short questions about [What is privacy?](/course/open/rai-what-is-privacy?lang=en) and [Saving and loading a model](/course/open/torch-checkpoints?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_taxonomy():
    quick_taxonomy = L.choice(
        "quick-taxonomy",
        question="What does Solove's taxonomy (2006) sort into groups?",
        options={
            "right": "Activities that can cause privacy problems",
            "definition": "Definitions of privacy, from narrow to broad",
            "data": "Kinds of personal data, by how sensitive they are",
        },
        correct="right",
        why={
            "right": "Solove sorts the activities that can cause privacy problems into four groups and sixteen kinds: collection, processing, dissemination and invasion. It is not a definition of privacy.",
            "definition": "Solove's taxonomy defines nothing: it sorts the activities that can cause privacy problems. See *Solove's taxonomy* in *The idea* of [What is privacy?](/course/open/rai-what-is-privacy?lang=en).",
            "data": "The taxonomy sorts what people and organizations do with information, such as collecting and spreading it, not the kinds of data. See *Solove's taxonomy* in *The idea* of [What is privacy?](/course/open/rai-what-is-privacy?lang=en).",
        },
    )
    quick_taxonomy
    return (quick_taxonomy,)


@app.cell(hide_code=True)
def quick_check_state():
    quick_state = L.choice(
        "quick-state",
        question="What does `model.state_dict()` of a `DigitsMLP(64)` give?",
        options={
            "copy": "A copy of the model, code and numbers together",
            "right": "A dictionary with every parameter by name",
            "loss": "Its loss after the last epoch",
        },
        correct="right",
        why={
            "right": "`model.state_dict()` is a dictionary with every parameter by name: `hidden.weight`, `hidden.bias`, `out.weight` and `out.bias`, each the model's own tensor.",
            "copy": "A state_dict holds only names and tensors; the code of the class stays in your own files. See *The state_dict* in *The idea* of [Saving and loading a model](/course/open/torch-checkpoints?lang=en).",
            "loss": "The state_dict holds the parameters of the model, not what its training measured. See *The state_dict* in *The idea* of [Saving and loading a model](/course/open/torch-checkpoints?lang=en).",
        },
    )
    quick_state
    return (quick_state,)


@app.cell(hide_code=True)
def quick_check_load():
    quick_load = L.choice(
        "quick-load",
        question="`model.load_state_dict(state)`, with `state` from another `DigitsMLP(64)`. What does it do?",
        options={
            "right": "It copies those tensors into `model`",
            "train": "It trains `model` until it matches `state`",
            "share": "It makes both models share one set of tensors",
        },
        correct="right",
        why={
            "right": "`load_state_dict` copies the tensors of a state_dict into a model of the same class and sizes. Each phone in this lesson does that with the server's weights before it trains.",
            "train": "Loading trains nothing: it copies the numbers of `state` into `model`, name by name. See *Save and load* in *The idea* of [Saving and loading a model](/course/open/torch-checkpoints?lang=en).",
            "share": "It copies the numbers into the model's own tensors: training `model` afterwards does not change the tensors in `state`. See *Save and load* in *The idea* of [Saving and loading a model](/course/open/torch-checkpoints?lang=en).",
        },
    )
    quick_load
    return (quick_load,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Your phone's keyboard guesses the next word you will type. To train such a model, a company could collect what millions of people type and keep it on its servers. H. Brendan McMahan and colleagues at Google proposed another way, in a paper published in 2017: "Each client has a local training dataset which is never uploaded to the server. Instead, each client computes an update to the current global model maintained by the server, and only this update is communicated." Andrew Hard and colleagues (2018) trained a next-word model of Gboard, Google's keyboard, that way, on 1.5 million phones. In this lesson you spread 6,000 handwritten digits over ten phones, let a server train one model on all of them without ever seeing a digit, and see what changes when each phone holds only a few kinds of digit.

    **An everyday example.** Ten bakers want to improve one bread recipe without letting anyone into their kitchens. Each tries the recipe at home, adjusts the amounts, and sends back only the adjusted recipe. The club averages the amounts, counting a baker who baked more loaves more heavily, and sends the new recipe round again.
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="McMahan et al. spread MNIST over 100 clients. When every client had a bit of every digit, their method took 18 rounds to train a CNN, a network for images that a later lesson covers, to 99% test accuracy. How many rounds did it take when most clients held only two digits?",
        options={
            "same": "About 18 as well",
            "right": "About 170 rounds",
            "never": "It never got there",
        },
        reveal="**About 170**: 173 rounds, against 18 when every client had a bit of every digit (McMahan et al. 2017, Table 2, with 20 epochs of mini-batches of 10 on each client per round). Training more on each client still saved rounds on that split, but far fewer than on the shuffled one; the authors call those gains \"smaller, but still substantial\". In the demo you see the same slowdown on ten phones.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### The data stay on the phone

    McMahan et al. (2017) gave the setting its name: "We term this decentralized approach Federated Learning." A phone, or another device that trains on its own data, is a **client**; the **server** holds the shared model. Peter Kairouz, McMahan and 57 more authors (2021) define it so: "Federated learning is a machine learning setting where multiple entities (clients) collaborate in solving a machine learning problem, under the coordination of a central server or service provider. Each client's raw data is stored locally and not exchanged or transferred". Apple used it in iOS 13, they write, for its QuickType keyboard and to recognize "Hey Siri".

    The data of clients are unlike the data of an ordinary training run. McMahan et al. name four ways: each client's data differ from the data as a whole, clients hold very different amounts, there are very many clients, and communication with them is limited. And the server is still there: "Clearly, some trust of the server coordinating the training is still required."

    ### FedAvg: average the weights

    McMahan et al. call their method Federated Averaging, **FedAvg**. It has three knobs: C, the share of the clients that train in each round; E, the number of epochs each client makes over its own data in a round; and B, the size of its mini-batches. A **round** goes like this. The server picks m = max(C · K, 1) of its K clients at random. Each loads the server's weights w<sub>t</sub>, trains E epochs of mini-batches of B on its own data, and sends back its new weights w<sup>k</sup><sub>t+1</sub>. The server takes their average, with each client counting by its number of examples n<sub>k</sub>:

    $$w_{t+1} = \sum_{k \in S_t} \frac{n_k}{m_t}\, w^k_{t+1}, \qquad m_t = \sum_{k \in S_t} n_k$$

    with S<sub>t</sub> the clients chosen in round t, and m<sub>t</sub> all their examples together: m counts clients, m<sub>t</sub> counts examples. A client with 600 examples counts six times as much as one with 100. The sum runs over the chosen clients only; the authors add: "Earlier versions of this paper incorrectly indicated summation over all K clients here."

    Why can you average the weights of two models at all? McMahan et al. found that it depends on where they start: "when we start two models from the same random initialization and then again train each independently on a different subset of the data (as described above), we find that naive parameter averaging works surprisingly well". From different random starts, averaging failed in their test. In FedAvg every client starts each round from the server's weights. Training more on each client before averaging saves rounds: over their models and datasets, they report "a reduction in required communication rounds by 10–100× as compared to synchronized stochastic gradient descent".

    ### Phones with different digits

    On real phones, each client's data look like its owner, not like everyone. McMahan et al. tested that on MNIST with 100 clients of 600 digits each, with an MLP and with a CNN. Shuffled, each client got a bit of every digit. Sorted, the digits were cut into 200 pieces of 300, and each client got two pieces, a split they call pathological, since most clients then "will only have examples of two digits". With 10 clients per round, E = 20 and B = 10, the CNN reached 99% test accuracy in 18 rounds on shuffled clients, and in 173 on sorted ones. Against averaging after a single step on all of a client's data, training more on each client cut the rounds 35 times for the CNN and 46 times for their MLP on shuffled clients, and only 2.8 to 3.7 times on sorted ones. How the data differ matters, though: on their Shakespeare data, with one client per speaking role in a play, FedAvg needed 41 rounds to reach its target, where the shuffled lines needed 192.

    ### What the weights still tell

    Federated learning keeps the raw data on the phone, but the weights a phone sends were learned from those data. McMahan et al. warn themselves that for a word model whose input says only which words occur, the gradient shows exactly which words a user typed; for a dense model such as a CNN it is harder, "though attacks are still possible". Kairouz, McMahan et al. write: "there is still no formal guarantee of privacy in this baseline federated learning model." Ligeng Zhu, Zhijian Liu and Song Han (2019) showed "that it is possible to obtain the private training data from the publicly shared gradients": from the gradients of one training step they rebuilt the training images, and "the recovery is pixelwise accurate for images". Their test had conditions: networks with sigmoid instead of ReLU, random weights and batches of at most 8 images. FedAvg sends weights after many local steps, a case their paper does not test. Noise on the gradients stopped the attack only once it hurt the model: in their test on CIFAR-100, the accuracy fell from 76.3% to 45.3%.

    Hard et al. (2018) trained Gboard's model with FedAvg, in 3,000 rounds of 100 to 500 phones each, and write: "The federated learning procedure discussed here requires users to trust that the aggregation server will not scrutinize individual weight uploads." On the server's logs, their model predicted the next word right 16.4% of the time, against 16.5% for the same model trained on the server; on the phones' own data, 15.8% against 15.0%. McMahan et al. point to stronger guarantees "via differential privacy [14, 13, 1], secure multi-party computation [18], or their combination".
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **federated learning** | training one shared model on data that stay on many devices: each trains on its own data and sends only what it learned to a server |
        | **client** | a phone or other device in federated learning, with its own data |
        | **server** | the computer that holds the shared model, picks the clients of a round and averages what they send |
        | **round** | one turn: the server sends its weights, the chosen clients train and send theirs back, and the server averages them |
        | **FedAvg** | Federated Averaging: each round, the chosen clients train E epochs from the server's weights, and the server averages their weights, each counting by its number of examples |
        | **weighted average** | an average in which each value counts more or less, here by its number of examples: Σ n<sub>k</sub> w<sub>k</sub> / Σ n<sub>k</sub> |
        | **C, E, B** | the share of the clients in a round, the epochs of each client in a round, and the size of its mini-batches |
        """),
        L.watch_out("Federated learning keeps the raw data on the phone, not all of the privacy: the weights a phone sends can give away something about its data, and the server must be trusted not to look at each phone's weights. And FedAvg averages only the clients chosen in that round, each counting by its number of examples."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To train one model on data that stay on the phones, with FedAvg:

    1. The server makes the model and keeps its weights, as a state_dict.
    2. Each round it picks some phones at random. Each loads the server's weights with `load_state_dict`, trains on its own data, and sends back only its new state_dict and its number of examples.
    3. For every name in the state_dict, the server averages the phones' tensors, each counting by its number of examples: `fedavg` does that for one number, `fedavg_rows` for every column of an array at once, and `fedavg_states` for a whole state_dict.
    4. The average becomes the server's weights, and the next round starts from there.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell spreads the 6,000 training digits of QMNIST over 10 phones of 600 each, in a random order, so that each phone holds a bit of every digit. `train_on_phone(state, k)` is phone k's part of a round: it loads the server's weights, trains one epoch over its own 600 digits in mini-batches of 10, and gives back the new weights. Two phones train from the same weights of the server, and the server takes their average. The cell measures each model on the 1,000 test digits. It takes a second.
    """)
    return


@app.cell
def example_round():
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    # 1. the 6,000 training digits, spread over 10 phones of 600 each
    data = np.load("data/qmnist-small.npz")
    X = torch.tensor(data["train_images"], dtype=torch.float32) / 255
    y = torch.tensor(data["train_labels"], dtype=torch.int64)
    X_test = torch.tensor(data["test_images"], dtype=torch.float32) / 255
    y_test = torch.tensor(data["test_labels"], dtype=torch.int64)
    phones = np.random.default_rng(0).permutation(6000).reshape(10, 600)


    class DigitsMLP(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.hidden = nn.Linear(784, hidden)
            self.out = nn.Linear(hidden, 10)

        def forward(self, x):
            return self.out(torch.relu(self.hidden(x)))


    def train_on_phone(state, k):
        # 2. on phone k: start from the server's weights, one epoch over its own 600 digits
        model = DigitsMLP(64)
        model.load_state_dict(state)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(X[phones[k]], y[phones[k]]), batch_size=10, shuffle=True)
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return model.state_dict()   # 3. only the weights go to the server


    def accuracy(state):
        model = DigitsMLP(64)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            return (model(X_test).argmax(dim=1) == y_test).float().mean().item()


    torch.manual_seed(0)
    server = DigitsMLP(64).state_dict()
    torch.manual_seed(2)
    a = train_on_phone(server, 0)
    b = train_on_phone(server, 1)
    # 4. the server averages: both phones hold 600 digits, so each counts half
    new = {}
    for name in a:
        new[name] = (a[name] + b[name]) / 2
    print("server, before:", round(accuracy(server), 3))
    print("phone 0:", round(accuracy(a), 3))
    print("phone 1:", round(accuracy(b), 3))
    print("average:", round(accuracy(new), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Before training, the server's model reads 135 of the 1,000 test digits right: little more than guessing, which gets about 100. After one epoch on its own 600 digits, phone 0's model reads 724 right and phone 1's 772. Their average, a model that nobody trained, reads 775: a little more than the better of the two. Both phones started from the same weights, `server`, which is when McMahan et al. found that averaging works. The server never saw a digit: only `a` and `b`, two state_dicts, reached it. Each comes from a new model of its own phone, so the server's `server` stays as it was: `load_state_dict` copies the numbers, and the state_dict of a model is that model's own tensors, as in [Saving and loading a model](/course/open/torch-checkpoints?lang=en). Both phones hold 600 digits, so each counts half. In the exercises you write the average for phones of any size: first for one number, then for many at once, then for whole state_dicts.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_fedavg():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. fedavg: one weight

        Three phones each send one weight of the same model: 0.2, 0.8 and 0.6. They trained on 100, 300 and 600 examples.

        `fedavg(weights, sizes)` gets two lists of the same length: `weights`, one weight from each phone, and `sizes`, each phone's number of examples. It gives back the weighted average: each weight times its phone's number of examples, added up, divided by the number of examples of all phones together. The first lines start `total` and `examples` at 0. Replace the first `...` with a loop over the phones, and the second with the result. The check calls `fedavg([0.2, 0.8, 0.6], [100, 300, 600])`, then other phones.
        """),
        L.task("Give back the average of weights, each counting by its number of examples in sizes."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fedavg():
    box_fedavg = L.code_box("fedavg", lines=8)
    box_fedavg
    return (box_fedavg,)


@app.cell(hide_code=True)
def check_fedavg(box_fedavg):
    result_fedavg = L.check("fedavg", box_fedavg.value)
    result_fedavg
    return (result_fedavg,)


@app.cell(hide_code=True)
def hints_fedavg():
    L.hints(
        "fedavg",
        [
            "Step 3 of the recipe, for one number: the formula in *FedAvg: average the weights*.",
            "Go over the phones with `for k in range(len(weights)):`, and add up two things in the loop: in `total`, phone k's weight, counted as often as phone k has examples; in `examples`, its number of examples. After the loop, divide the one by the other.",
            "`        total = total + ___`",
        ],
        why="In the loop, `total = total + sizes[k] * weights[k]` and `examples = examples + sizes[k]`; after it, `return total / examples`. That gives (100 · 0.2 + 300 · 0.8 + 600 · 0.6) / 1,000 = 620 / 1,000 = 0.62. The plain mean of the three weights is 0.53: the phone with 600 examples pulls the average toward its 0.6.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fedavg_rows():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. fedavg_rows: every weight at once

        A model has many weights, and a loop over each of them is slow. `ROWS` holds four weights of each of the three phones, one row per phone; its first column holds the weights of exercise 1. `SIZES` holds the phones' numbers of examples as a NumPy array: 100, 300 and 600.

        `fedavg_rows(W, sizes)` gets a 2D NumPy array `W`, with one row per phone and one column per weight, and a 1D NumPy array `sizes`, with each phone's number of examples. It gives back the weighted average of every column at once, as a 1D array. Replace the `...`, without a loop. The check calls `fedavg_rows(ROWS, SIZES)`, then other arrays.
        """),
        L.task("Give back the weighted average of every column of W, each row counting by its number of examples in sizes."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fedavg_rows():
    box_fedavg_rows = L.code_box("fedavg-rows", lines=6)
    box_fedavg_rows
    return (box_fedavg_rows,)


@app.cell(hide_code=True)
def check_fedavg_rows(box_fedavg_rows):
    result_fedavg_rows = L.check("fedavg-rows", box_fedavg_rows.value)
    result_fedavg_rows
    return (result_fedavg_rows,)


@app.cell(hide_code=True)
def hints_fedavg_rows():
    L.hints(
        "fedavg-rows",
        [
            "The same weighted average as in exercise 1, for every column at once: with the matrix product `@`.",
            "`v @ W`, with `v` a 1D array of length 3 and `W` an array of 3 rows, adds the rows up, row k times `v[k]`: one number for each column. That is the top of the fraction of exercise 1, for every column at once. The bottom is the same for every column.",
            "`    return sizes @ W / ___`",
        ],
        why="`return sizes @ W / np.sum(sizes)` gives `[0.62, 0.5, 3.0, 0.6]`: the first column is exercise 1 again, and the third, where every phone sends 3.0, stays 3.0.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fedavg_states():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. fedavg_states: a whole state_dict

        A phone sends its weights as a state_dict: a dictionary with a tensor for every name, such as `hidden.weight`. `STATES` is a list of three state_dicts of a tiny model with two tensors, `w` and `b`: one for each phone of exercise 1.

        `fedavg_states(states, sizes)` gets `states`, a list with one state_dict for each phone, and `sizes`, a list with each phone's number of examples. It gives back a new dictionary `new` with the same names: for every name, the weighted average of the phones' tensors of that name. The first lines add all examples up in `examples` and start `new` empty. Replace the `...` with a loop over the names in `states[0]`, with a loop over the phones inside it. The check calls `fedavg_states(STATES, [100, 300, 600])`, then state_dicts of `nn.Linear(3, 2)` and other phones.
        """),
        L.task("Give back a dictionary with, for every name, the average of the phones' tensors, each counting by its number of examples."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fedavg_states():
    box_fedavg_states = L.code_box("fedavg-states", lines=12)
    box_fedavg_states
    return (box_fedavg_states,)


@app.cell(hide_code=True)
def check_fedavg_states(box_fedavg_states):
    result_fedavg_states = L.check("fedavg-states", box_fedavg_states.value)
    result_fedavg_states
    return (result_fedavg_states,)


@app.cell(hide_code=True)
def hints_fedavg_states():
    L.hints(
        "fedavg-states",
        [
            "Step 3 of the recipe: the end of the worked example, for any number of phones, each counting by its number of examples.",
            "For each `name` in `states[0]`, start a `total` at 0 and go over the phones with `for k in range(len(states)):`, adding phone k's tensor of that name, counted as often as phone k has examples. A number times a tensor is a tensor, so `total` becomes one. After the loop over the phones, `new[name]` is `total` divided by all examples.",
            "`            total = total + ___`",
        ],
        why="Inside `for name in states[0]:`, `total = 0`, then `for k in range(len(states)):` with `total = total + sizes[k] * states[k][name]`, and after it `new[name] = total / examples`. On `STATES`, `w` becomes `[[0.62, 0.5], [3.0, 0.6]]` and `b` becomes `[0.6, 1.0]`: the numbers of `ROWS` in exercise 2, now name by name. In the picture, three phones with 600, 300 and 100 digits read 735, 555 and 406 test digits right; their weighted average reads 765, more than any of the three. A plain mean, with the phone of 100 digits counting as much as the one of 600, reads 723.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### FedAvg with your fedavg_states

    The button runs FedAvg twice, for 20 rounds each, with your `fedavg_states` as the server's step once exercise 3 is right, and the course's until then. Each round the server picks 5 of the 10 phones at random; each trains one epoch from the server's weights, as `train_on_phone` does in the worked example, and only the weights come back. The first run uses the phones of the worked example, with a bit of every digit. The second sorts the 6,000 digits by digit first, cuts them into 20 pieces of 300 and gives each phone two pieces at random, as McMahan et al. did with MNIST: each phone then holds two or three digits. While it trains, a chart shows after every round the share of the 1,000 test digits that the server's model reads wrong, and beside it what that model guesses for one test digit of each kind. It takes a few seconds. Afterwards you can go back through the rounds.
    """)
    return


@app.cell(hide_code=True)
def fedavg_button_cell():
    fedavg_button = mo.ui.run_button(label="Run FedAvg on both kinds of phones")
    fedavg_button
    return (fedavg_button,)


@app.cell(hide_code=True)
def demo_fedavg(box_fedavg_states, fedavg_button, result_fedavg_states):
    if not fedavg_button.value:
        _out = mo.md("*Press the button to start the two runs.*")
    else:
        _yours = C.mine(box_fedavg_states.value, result_fedavg_states.state == "correct")
        _kind = {"mixed": "a bit of every digit", "by_digit": "two or three digits each"}
        _now = {}

        def _progress(_name, _p):
            if _now.get("name") != _name:   # a new run: a new chart, with the run before it dashed
                _before = list(zip(_now["live"].xs, _now["live"].ys)) if "live" in _now else []
                _now["name"] = _name
                _now["live"] = L.live({"title": f"Phones with {_kind[_name]}", "x": "round", "y": "share of test digits wrong",
                                       "alt": f"The share of the 1,000 test digits that the server's model reads wrong after every round so far, on phones with {_kind[_name]}{'; dashed, the run on phones with a bit of every digit' if _before else ''}."},
                                      C.ROUNDS, low=0.0, high=1.0)
                if _before:
                    _now["live"].reference(_before, "every digit", role="loss")
            _now["live"].watch(lambda: C.now_picture(_p["said"], f"What the server's model guesses after round {_p['round']}", L.lang))
            _now["live"].add(_p["round"], _p["wrong"], f"**Phones with {_kind[_name]}** · round {_p['round']} of {_p['rounds']} · {round(1000 * (1 - _p['wrong']))} of 1,000 test digits right")

        mo.output.replace(mo.md("*Round 1 · starting …*"))
        _runs = C.demo(_yours or C.fedavg_states, _progress)
        if isinstance(_runs, str):
            _note = {"average": "*Your `fedavg_states` stops with an error on the phones' weights, or gives weights that `DigitsMLP(64)` cannot load. The picture under exercise 3 says what goes wrong.*",
                     "slow": "*The two runs take more than three minutes. Does your `fedavg_states` go over the names and the phones only once?*"}[_runs]
            _out = _now["live"].stopped(_note) if "live" in _now else mo.md(_note)
        else:
            _right = {_k: [round(1000 * (1 - _w)) for _w in _runs[_k][0]] for _k in _runs}
            _m, _d = _right["mixed"], _right["by_digit"]
            _best = max(range(len(_d)), key=lambda _i: _d[_i])
            _words = {"title": "Two runs of FedAvg", "x": "round", "y": "share of test digits wrong",
                      "mixed": "every digit", "by_digit": "two or three digits",
                      "alt": f"Two lines over 20 rounds: the share of the test digits read wrong falls to {L.number(1 - _m[-1] / 1000, 3)} with phones that hold a bit of every digit, dashed, and jumps about on its way to {L.number(1 - _d[-1] / 1000, 3)} with phones that hold two or three digits each.",
                      "guess_mixed": "Every digit: the guesses after round {r}", "guess_by_digit": "Two or three digits: the guesses after round {r}"}
            _who = "your `fedavg_states`" if _yours else "the course's `fedavg_states`, as exercise 3 is not right yet"
            _out = mo.vstack([
                mo.md(f"""With {_who}: after 20 rounds, the server's model reads <span class="ea-num">{_m[-1]}</span> of the 1,000 test digits right with phones that hold a bit of every digit, and <span class="ea-num">{_d[-1]}</span> with phones that hold two or three digits each. On the mixed phones it reads {_m[4]} right after 5 rounds already; on the sorted phones the line jumps from round to round, as each round's 5 phones hold other digits, and its best round, round {_best + 1}, reads {_d[_best]} right."""),
                L.explore("rewind", "round", lambda _r: (C.rewind_frame(_runs, int(_r), _words, L.lang),
                                                         f"After round {int(_r)}: {_m[int(_r) - 1]} test digits right with a bit of every digit on each phone, {_d[int(_r) - 1]} with two or three digits per phone."),
                          start=C.ROUNDS, values=list(range(1, C.ROUNDS + 1)), role="",
                          task="Go back to round 1, then forward again: which digits does the server's model of the sorted phones read right first?"),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_two_digits():
    mo.md(r"""
    ### Exercise 4. Phones that differ

    A keyboard learns with FedAvg from phones in many countries. In each country, people type mostly in their own language.
    """)
    return


@app.cell(hide_code=True)
def choice_two_digits_cell():
    choice_two_digits = L.choice(
        "two-digits",
        question="What does the demo suggest for FedAvg on these phones?",
        options={
            "same": "Just as fast: the average evens the countries out",
            "right": "Slower than on phones with a bit of everything",
            "never": "It cannot learn: no phone has all languages",
        },
        correct="right",
        why={
            "right": "In the demo, phones with two or three digits each gave a server model that learned more slowly and jumped from round to round: in the course's run, 678 test digits right after 20 rounds, against 904 with a bit of every digit on each phone. McMahan et al. saw the same on MNIST: 173 rounds to 99% instead of 18. Not every difference slows FedAvg down, though: on their Shakespeare data, split by speaking role, it needed fewer rounds than on shuffled lines.",
            "same": "In the demo, the average of phones that each hold a few digits learned more slowly than with mixed phones: in the course's run, 678 test digits right after 20 rounds, against 904. See *Phones with different digits* in *The idea*.",
            "never": "It still learns, only more slowly: in the course's run, the sorted phones reached 804 test digits right in their best round, and McMahan et al. reached 99% on sorted MNIST in 173 rounds. See *Phones with different digits* in *The idea*.",
        },
    )
    choice_two_digits
    return (choice_two_digits,)


@app.cell(hide_code=True)
def text_exercise_server_learns():
    mo.md(r"""
    ### Exercise 5. What the server sees

    In the demo, the server never got a digit: only state_dicts.
    """)
    return


@app.cell(hide_code=True)
def choice_server_learns_cell():
    choice_server_learns = L.choice(
        "server-learns",
        question="Can the server still learn something about a phone's digits?",
        options={
            "no": "No: the digits themselves never left the phones",
            "right": "Yes: the weights were learned from them",
            "labels": "Only if a phone also sends its labels",
        },
        correct="right",
        why={
            "right": "The weights were learned from the phone's digits, so they carry something of them. Zhu, Liu and Han rebuilt training images from the gradients of one step, and Kairouz, McMahan et al. write that the basic setting gives no formal guarantee of privacy. Hard et al. ask users to trust that the server will not look at each phone's weights.",
            "no": "The digits stayed, but the weights a phone sends were learned from them: Zhu, Liu and Han rebuilt training images from gradients. See *What the weights still tell* in *The idea*.",
            "labels": "What a phone sends can be enough: from the gradients of one step, Zhu, Liu and Han rebuilt training images and their labels, and nobody sent them either. See *What the weights still tell* in *The idea*.",
        },
    )
    choice_server_learns
    return (choice_server_learns,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [What is privacy?](/course/open/rai-what-is-privacy?lang=en), collection was the first group of Solove's taxonomy. Federated learning trains without collecting the data at all, though what the phones send still says something about them. Each phone's message is a state_dict, as in [Saving and loading a model](/course/open/torch-checkpoints?lang=en), and `load_state_dict` starts each phone from the server's weights. McMahan et al. name secure multi-party computation as a way to stronger guarantees: in the next lesson, *Computing on secrets*, several parties add up their numbers without anyone seeing another's.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_server_learns, choice_two_digits, result_fedavg, result_fedavg_rows, result_fedavg_states):
    L.wrap_up(result_fedavg, result_fedavg_rows, result_fedavg_states, choice_two_digits, choice_server_learns)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
