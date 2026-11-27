# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-reproducibility. Route: course/route/lessen/systemen.yaml. Checks, the course's runs, the demo and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-reproducibility", "en")
    C = L.checks   # the course's runs of the digits MLP, the demo and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Random numbers and seeds](/course/open/np-random?lang=en) and [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rolls():
    quick_rolls = L.choice(
        "quick-rolls",
        question="Two random number generators are both made with `np.random.default_rng(7)`. Each gives three dice rolls, and then the first one gives three more. Which rolls are the same?",
        options={
            "all": "All nine, since both have the same seed",
            "right": "The first three of each generator",
            "none": "None: every roll is random",
            "last": "The last three of the first and the three of the second",
        },
        correct="right",
        why={
            "right": "Each generator starts at its seed, so both give the same first three rolls, and then the first one goes on to new rolls. A training run gets its random numbers the same way: the same seed gives the same numbers, in the same order.",
            "all": "The first generator goes on after three rolls, to new numbers. Which of its rolls can the second one match? See [Random numbers and seeds](/course/open/np-random?lang=en).",
            "none": "The numbers look random, but they follow from the seed. What do two generators with seed 7 give? See [Random numbers and seeds](/course/open/np-random?lang=en).",
            "last": "The last three of the first generator come after its first three, so they are new. Which rolls match? See [Random numbers and seeds](/course/open/np-random?lang=en).",
        },
    )
    quick_rolls
    return (quick_rolls,)


@app.cell(hide_code=True)
def quick_check_weights():
    quick_weights = L.choice(
        "quick-weights",
        question="Before any training, `DigitsMLP(64)` already gives ten logits for a digit. Where do its weights come from?",
        options={
            "zeros": "All zero, until the first step",
            "data": "From the training digits, averaged per label",
            "right": "From PyTorch's random number generator",
            "file": "From a file the course ships with the lesson, the same for every learner",
        },
        correct="right",
        why={
            "right": "`nn.Linear` starts its weights as small random numbers from PyTorch's generator. So `torch.manual_seed(0)` before the model gives the same start every time, as in the project. This lesson adds the order of the examples, and what else a run needs to come out the same twice.",
            "zeros": "With all weights zero, every hidden neuron would compute the same and learn the same. What does `nn.Linear` start with? See [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en).",
            "data": "Training brings in the digits; before it, the weights know nothing of them. What do they start as? See [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en).",
            "file": "The course ships the digits, not the weights: `DigitsMLP(64)` makes them itself. From what? See [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en).",
        },
    )
    quick_weights
    return (quick_weights,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In 2018, Peter Henderson and colleagues trained one reinforcement learning method ten times on the same task, a simulated robot that learns to run, with the same code and the same settings. Only the seed changed. They split the ten runs into two groups of five seeds, and a statistical test called the two groups different, as if two different methods had been compared. Their conclusion: "the variance between runs is enough to create statistically different distributions just from varying random seeds." A paper that compares one run of its own method with one run of another may be measuring luck.

    The other side matters as much. When a run goes wrong, say its loss jumps up in epoch 7, you can only find out why if you can run it again and see the same jump. This lesson is about both: making a run come out the same again, and seeing how far apart runs with other seeds land.

    **An everyday example.** One race says little about which of two runners is faster: on another day, with other wind, the order can flip. A recording of the race shows the same finish every time you play it, but it is still one race. To know who is faster, you need several races.
    """)
    return


@app.cell(hide_code=True)
def predict_spread():
    guess_spread = L.guess(
        "spread",
        question="You train the digits MLP from the project ten times: the same code, the same data and the same settings, with the seeds 0 to 9. How far apart do the best and the worst run land, in test digits right of 1,000?",
        options={
            "none": "Not at all: the same code gives the same result",
            "two": "About 2 digits",
            "right": "About 15 digits",
            "hundred": "About 150 digits",
        },
        reveal="**About 15 digits.** With the seeds 0 to 9, the best run reads 935 test digits right and the worst 919: 16 apart, with the same code, data and settings. Only the random numbers changed: the weights at the start and the order of the examples. Run one seed again, and you get its number again. The demo below shows all ten runs.",
    )
    guess_spread
    return (guess_spread,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Where the random numbers come in.** The training of the digits MLP uses random numbers in two places, both from [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en): the weights of `DigitsMLP` start as small random numbers, and a DataLoader with `shuffle=True` puts the examples in a new random order every epoch. Both come from PyTorch's random number generator, which `torch.manual_seed` gives a seed, as `np.random.default_rng` takes one in [Random numbers and seeds](/course/open/np-random?lang=en). With the same seed, a run starts from the same weights and sees the examples in the same order.

        **A generator of its own.** `torch.manual_seed` seeds PyTorch's global generator: one generator for the whole program, and every random number comes from it in turn. Add one random call before training, such as a look at a few random examples, and every number after it shifts: other weights, another order. A generator of its own keeps one source apart. `rng = torch.Generator()` makes one, and `rng.manual_seed(1)` gives it a seed. `DataLoader(ds, batch_size=64, shuffle=True, generator=rng)` takes its orders from it: every epoch, it asks `torch.randperm` for a random order of the positions of the examples, from that generator. With one generator for the weights and another for the order, you can change one seed and keep the other.

        **Determinism.** With the same seed, the same versions and the same hardware, training the digits MLP on a CPU gives the same model, bit for bit: that is **determinism**. Floats make it harder than it sounds. A float holds a limited number of digits, so an addition often has to round, and the order of the additions changes the rounding: the worked example below adds 0.1, 0.2 and 0.3 in two orders and gets two different numbers. A GPU, a chip that does thousands of calculations at the same time (*Why AI uses GPUs, and the same code on yours* explains it), splits a long addition into many small parts, and some of its operations add the parts in whatever order they finish. NVIDIA's documentation for cuDNN, the library PyTorch uses for many GPU operations, says such routines "do not guarantee reproducibility across runs, even on the same architecture" (the same kind of GPU). A difference in the last digit grows over thousands of training steps into a different model. `torch.use_deterministic_algorithms(True)` makes PyTorch use deterministic versions, or stop with an error where it has none; that often costs speed. Even then, PyTorch's documentation warns that results are not guaranteed to be the same "across PyTorch releases, individual commits, or different platforms", nor between CPU and GPU, "even when using identical seeds".

        **Write it down with the run.** A run you want to repeat needs its seed, its versions, which the lock file from [The same packages everywhere: environments and pinning](/course/open/sys-environments?lang=en) records, and its hardware, written down next to its result. Together they give **reproducibility**: someone else, or you next month, can run it again and get the same result. The picture shows what a run depends on.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **Run-to-run variance.** A run you can repeat is still one run. Picard (2021) trained one small image network with 10,000 different seeds: its test accuracy went from 89.01% to 90.83%, a difference the paper says is "widely considered as significant in the community". How much a result moves when only the seed changes is the **run-to-run variance**. To compare two settings, train each with several seeds and look at the spread: the worst run, the best run and the mean. One strict rule: call setting B better than setting A only when the spreads do not overlap, with every run of B above every run of A. Statistics can say more from fewer runs; this lesson keeps to the strict rule.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **reproducibility** | | how easily you, or someone else, can run your work again and get the same result | |
        | **determinism** | | the same code on the same data gives exactly the same result every time | |
        | **run-to-run variance** | | how much a result moves when you run the same experiment again with only another seed | |
        | **`torch.Generator()`** | | a random number generator of its own, apart from the one `torch.manual_seed` seeds | `rng = torch.Generator()` |
        | **`torch.randperm(n)`** | rand perm | the numbers 0 to n − 1 in a random order, as a tensor | `torch.randperm(5)` |
        """),
        L.watch_out("In Python, a generator is also a function with `yield` that hands out values one at a time. `torch.Generator` is something else: a random number generator."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    A run you can repeat, and a comparison you can trust:

    1. At the start of the run, give PyTorch a seed with `torch.manual_seed(seed)`, before you make the model.
    2. Give the order of the examples a generator of its own, with a seed, and pass it to the DataLoader with `generator=`.
    3. Write the seed, the versions (the lock file) and the hardware down next to the result.
    4. On a GPU, when runs must be the same bit for bit, add `torch.use_deterministic_algorithms(True)`, and follow PyTorch's page *Reproducibility* for the settings your GPU needs.
    5. To compare two settings, train each with several seeds, such as five or ten, and look at the worst run, the best run and the mean.
    6. Call a change better only when its spread lies above the old one.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    First the same three floats, added in two orders; then step 1 of the recipe, twice, with `torch.rand(3)`: three random floats between 0 and 1. `torch.equal` says whether two tensors hold exactly the same numbers. Run the cell.
    """)
    return


@app.cell
def example_seeds():
    import torch

    # floats: the same three numbers, added in two orders
    print((0.1 + 0.2) + 0.3)
    print(0.1 + (0.2 + 0.3))

    # 1. a seed: the same random numbers again
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
    The first two lines differ in the last digit: the same three numbers, added in another order. On the CPU of one computer, PyTorch adds in the same order on every run; a GPU can change the order from one run to the next. `first` and `again` hold the same three numbers, because the generator started over at seed 0 before each. `later` comes after them from the same generator, so it gets new numbers. A run that draws its random numbers from the same seed, in the same order, gets the same numbers every time.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_orders():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A new order every epoch, the same orders every run

        A DataLoader with `shuffle=True` and a generator of its own draws a new order every epoch. `orders(n, epochs, seed)` works the same way for `n` examples. `g = torch.Generator()` makes the generator, and `g.manual_seed(seed)` gives it the seed, once, before the loop. The loop `for epoch in range(epochs)` runs once per epoch. In it, `order` should get a random order of the numbers 0 to `n` − 1 from `torch.randperm`, drawn from `g`; `order.tolist()` turns the tensor into a list, and `found.append` adds that list to `found`. Replace the `...`.

        The check calls `orders(8, 2, 0)`, then other numbers of examples, epochs and seeds. The picture under it calls your function three times.
        """),
        L.task("Draw each epoch's order from the generator `g`."),
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
            "Step 2 of the recipe: the order gets a generator of its own, `g`. `torch.randperm(n)` gives the numbers 0 to `n` − 1 in a random order, as a tensor, and `order` keeps it.",
            "With only `n`, `torch.randperm` draws from PyTorch's global generator, and `g` stays unused. Which value with a name hands it a generator, as with a DataLoader?",
            "`        order = torch.randperm(n, ___)`",
        ],
        why="`torch.randperm(n, generator=g)` draws each order from `g`, which goes on from one epoch to the next: a new order every epoch. On the next run, `g.manual_seed(seed)` starts it at the same place again, so the orders come back the same.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Ten seeds, one MLP

    The course trained the digits MLP from the project ten times, with the seeds 0 to 9, and ten times more with 128 neurons in its hidden layer instead of 64: 10 epochs each, on a CPU, with PyTorch 2.14.1, the version in the course's lock file. The slider picks a seed. At the top, the loss of every epoch for all ten seeds with 64 neurons, with the chosen seed drawn thick in the color of the loss; below it, the number of test digits read right in all twenty runs, with the runs of the chosen seed drawn larger, in the color of the text, and the labels the first mini-batch of that seed starts with.
    """)
    return


@app.cell(hide_code=True)
def demo_seed():
    L.explore("seed", "seed", lambda _v: C.frame(_v, L.lang), start=0, low=0, high=9, step=1,
              role="", task="Find a seed where 64 neurons beat 128 neurons, and one where 128 neurons win.")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    The loss curves almost overlap, and still the ten runs with 64 neurons read from 919 to 935 test digits right. With 128 neurons it is 920 to 938. With five seeds 64 neurons win, with the other five 128 neurons. The means, 927.8 and 929.8, are 2 digits apart, far less than the 16 between the best and the worst run with 64 neurons. With one seed each, you could have concluded either.

    Where does the spread come from? The course trained 64 neurons again with two generators. Changing only the seed of the weights, with the order always from seed 0, gave 918 to 931; changing only the seed of the order gave 920 to 932. Each source alone gives most of the spread, so a seed for one and not for the other does not give you the same run.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_better():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Better, or a lucky seed?

        `better(a, b)` gets two lists with the results of several seeds: the number of test digits read right with setting A, and with setting B. It gives back `True` only when every run in `b` beats every run in `a`, and `False` otherwise. Replace the `...` with one comparison.

        The check calls `better(C.RUNS_64, C.RUNS_128)`, with the ten runs of the demo, then other lists of results.
        """),
        L.task("Say `True` only when every run of B beats every run of A."),
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
            "Every run of `b` beats every run of `a` when even the worst run of `b` does. Python's `min` gives the smallest number in a list, so `min(b)` is that worst run. Which run of `a` is the hardest to beat?",
            "The hardest run of `a` to beat is its best one, the largest number in the list. Which built-in function gives that?",
            "`    return min(b) > ___`",
        ],
        why="`min(b) > max(a)` is `True` only when the worst run of `b` beats the best run of `a`: then every run of `b` beats every run of `a`. For `C.RUNS_64` and `C.RUNS_128`, it compares 920 with 935, so it gives `False`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_gpu():
    mo.md(r"""
    ### Exercise 3. The same seed on a GPU

    A colleague trains your digits MLP on a GPU, with the same code, the same seed and the same lock file, and gets 929 test digits right instead of your 932 on a CPU. A second run on the same GPU gives 930.
    """)
    return


@app.cell(hide_code=True)
def choice_gpu_cell():
    choice_gpu = L.choice(
        "gpu",
        question="What is the most likely cause?",
        options={
            "seed": "The seed works on a CPU, not on a GPU",
            "lock": "The lock file installed other versions there",
            "right": "The GPU adds numbers in a changing order",
            "test": "The test digits come in another order on a GPU, so the count changes",
        },
        correct="right",
        why={
            "right": "Two runs on the same GPU already differ, so the cause is something that changes from run to run: the order in which the GPU adds numbers. `torch.use_deterministic_algorithms(True)` makes the GPU runs the same as each other, often at a cost in speed; they still need not match your 932 from a CPU.",
            "seed": "`torch.manual_seed` seeds the global generators of the CPU and of the GPU alike. And why would two runs on the same GPU differ?",
            "lock": "The lock file installs the same versions everywhere, and two runs on the same computer differ too. What changes from one run to the next?",
            "test": "The order of the test digits does not change how many the model reads right. What else changes from run to run?",
        },
    )
    choice_gpu
    return (choice_gpu,)


@app.cell(hide_code=True)
def text_exercise_one_seed():
    mo.md(r"""
    ### Exercise 4. One run against one run

    A colleague changes the learning rate of the digits MLP from 0.5 to 0.4, trains once with seed 0, and gets 935 test digits right instead of 932.
    """)
    return


@app.cell(hide_code=True)
def choice_one_seed_cell():
    choice_one_seed = L.choice(
        "one-seed",
        question="What do you suggest?",
        options={
            "keep": "Keep 0.4: three more digits right is better",
            "right": "Train both learning rates with ten seeds each, and compare",
            "again": "Train 0.4 again with seed 0, to be sure",
            "pick": "Try seeds for 0.4 until one clearly beats 932, and report that one",
        },
        correct="right",
        why={
            "right": "Three digits is well within the spread of ten seeds, so one run of each cannot tell the two apart. With ten seeds each, you see whether the spreads of 0.4 and 0.5 overlap.",
            "keep": "With other seeds, the runs of the old setting alone lay 16 digits apart. Can three digits from one seed tell the two apart?",
            "again": "With the same seed and the same versions, the run comes out the same: 935 again. What does a second run with the same seed not show?",
            "pick": "Then you report the luckiest seed, and the old setting gets no such chance. What would compare the two the same way?",
        },
    )
    choice_one_seed
    return (choice_one_seed,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [The same packages everywhere: environments and pinning](/course/open/sys-environments?lang=en) made the versions the same on every computer; this lesson adds the seeds, the order of the examples and the hardware. [Random numbers and seeds](/course/open/np-random?lang=en) gave the seed, and [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en) the model the course trained many times here. In the next lesson, *Keeping track of your runs: experiment tracking*, every run writes its seed, its settings and its results to a log, so you can compare runs later.
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
