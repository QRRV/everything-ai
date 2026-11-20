# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-reliability. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    from everything_ai.lesson import lesson

    L = lesson("rai-reliability", "en")
    C = L.checks   # the course's softmax regression for digits, the 1,000 test digits, three made-up agents


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Kinds of AI risk](/course/open/rai-kinds-of-ai-risk?lang=en) and [Project: a linear classifier for handwritten digits](/course/open/ml-digits-linear?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_category():
    quick_category = L.choice(
        "quick-category",
        question="A digit reader at a post office misreads smudged postcodes. Nobody meant harm, and it was meant to read them right. Where does the International AI Safety Report of 2025 put this?",
        options={
            "right": "Under malfunctions, with reliability issues",
            "misuse": "Under malicious use, as the harm falls on the senders",
            "systemic": "Under systemic risks, as nobody meant to cause harm",
        },
        correct="right",
        why={
            "right": "Nobody meant to cause harm, and the system did not do what was intended: a malfunction. The report of 2025 puts reliability issues there, next to bias and loss of control.",
            "misuse": "Malicious use needs someone who means to cause harm, and here nobody did. See [Kinds of AI risk](/course/open/rai-kinds-of-ai-risk?lang=en).",
            "systemic": "That nobody meant harm rules out malicious use; then ask whether the system did what was intended. It did not, so it is a malfunction. See [Kinds of AI risk](/course/open/rai-kinds-of-ai-risk?lang=en).",
        },
    )
    quick_category
    return (quick_category,)


@app.cell(hide_code=True)
def quick_check_logits():
    quick_logits = L.choice(
        "quick-logits",
        question="Softmax regression gets an image as a row `x` of 784 pixels. How does it choose a digit?",
        options={
            "common": "The digit that is most common in the training set",
            "right": "The digit with the largest of the ten logits `x @ W.T + b`",
            "bias": "The digit whose own bias term in `b` is the largest of the ten",
        },
        correct="right",
        why={
            "right": "Each digit has its own row of weights in `W` and its own bias term in `b`; the largest logit wins, and softmax only turns the logits into probabilities.",
            "common": "That is the baseline, which always says the same digit and reads 126 of 1,000 right. See [Project: a linear classifier for handwritten digits](/course/open/ml-digits-linear?lang=en).",
            "bias": "The bias terms are only part of each logit; the pixels times the weights decide most of it. See [Project: a linear classifier for handwritten digits](/course/open/ml-digits-linear?lang=en).",
        },
    )
    quick_logits
    return (quick_logits,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In the project lesson, softmax regression read 890 of 1,000 new digits right. Those digits were scanned the same way as the 6,000 it learned from. In use, a scan may have a gray haze over it or noise from a cheap scanner, or a digit may be written a little to one side. Does the model still read them?

    Keeping working takes more than a good score on a test set. This lesson gives four words for it, measures two of them, on the digit classifier and on made-up agents, and shows how a law and a contract write them down.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="The image classifier AlexNet names 43.5% of clean ImageNet test photos wrong, so 56.5% right. A study added Gaussian noise to the photos, the grain of a picture taken in low light, at five strengths. Averaged over the five, what share does AlexNet name right?",
        options={
            "half": "About 50%: noise hardly matters",
            "right": "About 11%: one photo in nine",
            "third": "About 35%: a third of its hits lost",
        },
        reveal="**About 11%: one photo in nine.** AlexNet's error went from 43.5% on clean photos to 88.6% with Gaussian noise, averaged over the five strengths, so it named 11.4% right. People, the authors write, are not confused by many such corruptions: *Noise, blur and a shift* in *The idea* comes back to it.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Four words for keeping working

    In January 2023, the American standards institute NIST published its AI Risk Management Framework. Among the characteristics of AI systems that people can trust, it lists "valid and reliable" and "secure and resilient", and it bases its definitions on the international standard ISO/IEC TS 5723:2022:

    - reliability: the "ability of an item to perform as required, without failure, for a given time interval, under given conditions";
    - **robustness**: the "ability of a system to maintain its level of performance under a variety of circumstances". NIST adds that a robust system should also "perform in ways that minimize potential harms to people if it is operating in an unexpected setting";
    - resilience: in NIST's shorter words, "the ability to return to normal function after an unexpected adverse event"; a resilient system may also "degrade safely and gracefully when this is necessary".

    Availability appears in the framework as part of security: a secure system keeps its "confidentiality, integrity, and availability". The framework does not use the word predictability at all. This lesson takes it to mean doing the same thing each time the same task comes, and measures it with pass^k, below.

    NIST also warns that accuracy and robustness "can be in tension with one another". Accuracy on a test set tells how a model does on inputs like those in the test set, and no more.

    ### Noise, blur and a shift

    In 2019, Dan Hendrycks and Thomas Dietterich published ImageNet-C: the test photos of ImageNet with "15 types of algorithmically generated corruptions from noise, blur, weather, and digital categories", each at five strengths, 75 in all. The first is Gaussian noise, which "can appear in low-lighting conditions". Models are tested on these photos and "should not be trained on these images". The corruptions imitate ones that happen by themselves, "not worst-case adversarial perturbations": robustness against attacks is a subject of its own, which NIST and the AI Act put under security. "Humans are also not confused by many forms of corruption such as snow, blur, pixelation, and novel combinations of these", they write, while for the networks "from AlexNet to ResNet, corruption robustness in itself has barely changed. Thus our ‘superhuman’ classifiers are decidedly subhuman." In a second test of the same paper, with photos that change step by step, for example by a shift, a rotation or a zoom, a ResNet-18 changed its top answer between two neighboring frames of a zoom 15.6% of the time.

    ### What the AI Act asks

    The EU's AI Act, in its text of 2024, asks in Article 15 that high-risk AI systems "achieve an appropriate level of accuracy, robustness, and cybersecurity, and that they perform consistently in those respects throughout their lifecycle". They must be "as resilient as possible regarding errors, faults or inconsistencies", and robustness "may be achieved through technical redundancy solutions, which may include backup or fail-safe plans": a second system that takes over, or a safe way to stop. Attacks get a paragraph of their own: systems must be "resilient against attempts by unauthorised third parties" to change their use or output.
    """)
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### The same thing, every time

    An AI agent that changes orders or books flights should not succeed one day and fail the next on the same request. In 2024, Shunyu Yao and colleagues at the company Sierra built τ-bench, in which an agent helps a simulated customer of a shop (τ-retail) or of an airline (τ-airline). They proposed **pass^k** (read: pass hat k): "the chance that all k i.i.d. task trials are successful, averaged across tasks", where i.i.d. means that each try is independent of the others and made under the same conditions. A task that succeeds with chance p succeeds k times in a row with chance pᵏ, p times itself k times. Between tries, only what the language models wrote changed: "the user prompt and database transitions are the same". gpt-4o solved about 61% of the τ-retail tasks on one try. "With increasing k, the chance of consistently solving a task drops rapidly, to as low as ∼25% for pass^8 on τ-retail for the same model."

    ### There when needed

    Availability is measured in two ways, Google's book on site reliability engineering explains: as "the proportion of system uptime", or, as Google does, "in terms of the request success rate". A cloud contract for AI models, Amazon Bedrock's, takes the second: "‘Availability’ is calculated for each 5-minute interval as the percentage of Requests processed by Amazon Bedrock that do not fail with Errors", and an Error "is any Request that returns a 500 error code". The later lesson *Nines of reliability* works with such percentages.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **robustness** | keeping the level of performance when the input or the setting is a little different from the tests, such as noise, blur or a shift |
        | reliability | working as required, without failure, for a given time and under given conditions |
        | resilience | withstanding an unexpected event or change and returning to normal, or degrading safely |
        | availability | being there when needed: the share of the time, or of the requests, in which a system works |
        | predictability | doing the same thing each time the same task comes |
        | **pass^k** | the chance that all k tries at a task succeed, averaged over the tasks; pass^1 is the plain success rate |
        | pᵏ | p to the power k: p times itself, k times; Python writes a power with `**`, so `0.5 ** 3` is 0.125 |
        | `rng.normal(0, sigma, size=X.shape)` | random numbers with mean 0 and standard deviation `sigma`, one for each pixel of `X`: Gaussian noise |
        | `C.model()`, `C.test_set()` | the course's softmax regression, `W` and `b`, trained as in the project lesson (100 steps, learning rate 1, the same 6,000 digits), and the 1,000 test digits `X` with their labels `y` |
        | `C.AGENT`, `C.STEADY`, `C.SPLIT` | three made-up agents: each a list with the success rate of each of its ten tasks |
        """),
        L.watch_out("A model that is right on clean test digits can still fail on digits with a haze, noise or a shift. Test it on the changes you expect in use; robustness against attacks needs tests of its own."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Say which kind of keeping working you mean: robust against changed input, resilient after a failure, available when needed, or predictable from try to try.
    2. For robustness, choose changes that happen by themselves in use, such as noise, a haze or a shift, and keep them out of the training.
    3. Measure the accuracy with and without each change, at a few strengths.
    4. For predictability, run each task several times and compute pass^k, not only pass^1.
    5. For availability, read what counts as a failure: an error code, a timeout, or a wrong answer.
    6. Test robustness against attacks apart: NIST and the AI Act put it under security.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell adds 0.3 to every pixel of the 1,000 test digits, a gray haze over each digit, and counts the share the course's model still reads right, beside the clean digits.
    """)
    return


@app.cell
def example_haze():
    W, b = C.model()                                         # the course's softmax regression
    X, y = C.test_set()                                      # the 1,000 test digits and their labels
    for change in [0, 0.3]:
        predicted = np.argmax((X + change) @ W.T + b, axis=1)   # the digit with the largest logit
        print(change, np.mean(predicted == y))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Clean, the model reads 890 of the 1,000 right; with the haze, 741. The haze moves no stroke: every digit is where it was, on a gray ground. In Exercise 1 you add noise instead, of any strength.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_noisy():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. noisy_accuracy: the accuracy under noise

        Write `noisy_accuracy(W, b, X, y, sigma)`. Make a generator `rng` with `np.random.default_rng(0)`, so the noise is the same every run. Make `noisy`: `X` plus noise from `rng.normal`, with mean 0, standard deviation `sigma` and `size=X.shape`. Let `predicted` be the digit with the largest logit for every row of `noisy`, as in the worked example, and return the share of `predicted` that equals `y`. Replace the `...`. The check calls `noisy_accuracy(W, b, X, y, 0.5)` with the course's model and test digits, then other strengths and a small model of its own.
        """),
        L.task("The share of the noisy digits the model still reads right."),
    ])
    return


@app.cell(hide_code=True)
def exercise_noisy():
    box_noisy = L.code_box("noisy_accuracy", lines=8)
    box_noisy
    return (box_noisy,)


@app.cell(hide_code=True)
def check_noisy(box_noisy):
    result_noisy = L.check("noisy_accuracy", box_noisy.value)
    result_noisy
    return (result_noisy,)


@app.cell(hide_code=True)
def hints_noisy():
    L.hints(
        "noisy_accuracy",
        [
            "Three steps: make the noise and add it, let the model choose a digit for each noisy image, and count how often it is right.",
            "`rng.normal(0, sigma, size=X.shape)` gives one number per pixel. The logits are the noisy images times the transposed weights, plus the bias terms; `np.argmax` with `axis=1` takes the largest in each row, and `np.mean` of `predicted == y` gives the share.",
            "`    predicted = np.argmax(___, axis=1)`",
        ],
        why="`rng = np.random.default_rng(0)`, `noisy = X + rng.normal(0, sigma, size=X.shape)`, `predicted = np.argmax(noisy @ W.T + b, axis=1)` and `return np.mean(predicted == y)`. With noise of 0.5, the model reads 820 of the 1,000 test digits right.",
    )
    return


@app.cell(hide_code=True)
def text_after_noisy():
    mo.md(r"""
    With noise of 0.25 the model reads 874 of the 1,000 right, with 0.5 it reads 820, and with 1.0 still 637, although noise with a standard deviation of 1.0 is as large as the whole range of a pixel. A haze of 0.3 cost more than noise of 0.5. Every logit is a sum over the 784 pixels. The haze adds the same amount to it for every image: 0.3 times the sum of that digit's weights: +1.67 for a 2, but −3.41 for a 1, the lowest of the ten, so the model says a 1 only 22 times instead of 127. Noise around 0 moves each logit by about 1 for each image, but up as often as down: no digit loses on every image. That is this model on these digits; AlexNet lost far more to the Gaussian noise of ImageNet-C, on another task. Which change hurts depends on the model.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pass():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. pass_hat: all k tries

        `C.AGENT` is a made-up agent: a list with the success rate of each of its ten tasks, from 1.0 down to 0.0. Write `pass_hat(rates, k)`. For each rate `p` in `rates`, the chance that all `k` tries succeed is p to the power `k`. Add those up in `total`, which starts at 0, and return `total` divided by the number of tasks. Replace the `...`. The check calls `pass_hat(AGENT, 8)`, then other agents and values of `k`.
        """),
        L.task("The chance that all k tries succeed, averaged over the tasks."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pass():
    box_pass = L.code_box("pass_hat", lines=5)
    box_pass
    return (box_pass,)


@app.cell(hide_code=True)
def check_pass(box_pass):
    result_pass = L.check("pass_hat", box_pass.value)
    result_pass
    return (result_pass,)


@app.cell(hide_code=True)
def hints_pass():
    L.hints(
        "pass_hat",
        [
            "One task at a time: the chance that it succeeds `k` times in a row. Then the average over the tasks.",
            "Python writes a power with `**`. Add up in a `for` loop over `rates`, and divide by `len(rates)`.",
            "`        total = total + ___`",
        ],
        why="`total = 0`, `for p in rates:` with `total = total + p ** k` inside it, then `return total / len(rates)`. For AGENT with k = 8: 0.268.",
    )
    return


@app.cell(hide_code=True)
def text_after_pass():
    mo.md(r"""
    All three agents succeed on 60% of their tries: pass^1 is 0.6 for each. At k = 8, STEADY, with every task at 60%, falls to 0.017; AGENT falls to 0.268; SPLIT, which always solves six tasks and never the other four, stays at 0.6. The same success rate can hide very different consistency. The course made AGENT up to fall about as fast as gpt-4o on τ-retail, from about 61% to about 25%.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: a shift to the right

    The slider moves the 1,000 test digits a number of pixels to the right; the columns that come in on the left are black. The picture shows the first eight test digits with what the course's model says for each, and how many of the 1,000 it reads right at every shift.
    """)
    return


@app.cell(hide_code=True)
def demo_shift():
    shift = L.explore("shift", "pixels to the right", lambda _v: C.demo_frame(_v, L.lang),
                      start=0, values=C.SHIFTS, role="",
                      task="Find the smallest shift on the slider at which the model reads fewer than half of the test digits right.")
    shift
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    From 3 pixels on, the model reads fewer than half: 452 of the 1,000. One pixel costs 46 digits, two pixels 251, about as much as noise of 1.0. Each weight of softmax regression belongs to one pixel, so a stroke that moves lands on other weights: move the weights three pixels to the right as well, and the model reads 890 again. For a person, a digit two pixels to the side is the same digit.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_test():
    mo.md(r"""
    ### Exercise 3. Which test?

    The digit classifier will read postcodes on letters. A team wants to know how robust it is against changes that happen by themselves, as Hendrycks and Dietterich test it.
    """)
    return


@app.cell(hide_code=True)
def choice_test_cell():
    choice_test = L.choice(
        "test",
        question="Which test measures that?",
        options={
            "right": "Add noise, a haze or a small shift, and measure the accuracy again",
            "attack": "Change pixels on purpose, step by step, until the model reads a wrong digit",
            "repeat": "Show the same clean test digits again and check that every answer stays the same",
        },
        correct="right",
        why={
            "right": "Changes that happen by themselves, at a few strengths, measured against the clean accuracy: that is how Hendrycks and Dietterich test, with 15 kinds of corruption and, in their second test, shifts and zooms.",
            "attack": "That is an attack. ImageNet-C leaves out \"worst-case adversarial perturbations\", and NIST and the AI Act put attacks under security. See *Noise, blur and a shift* in *The idea*.",
            "repeat": "That measures whether the model does the same thing each time, which is predictability; softmax regression always gives the same answer for the same image. Robustness is about inputs that differ. See *Four words for keeping working* in *The idea*.",
        },
    )
    choice_test
    return (choice_test,)


@app.cell(hide_code=True)
def text_exercise_uptime():
    mo.md(r"""
    ### Exercise 4. Available, but right?

    For a whole month, a cloud service with an AI model answers every request within a second, never with a 500 error code. One answer in ten is wrong.
    """)
    return


@app.cell(hide_code=True)
def choice_uptime_cell():
    choice_uptime = L.choice(
        "uptime",
        question="In a contract like Amazon Bedrock's, what is the service's availability that month?",
        options={
            "ninety": "90%: each wrong answer counts as a failed request",
            "right": "100%: a wrong answer that arrives is not an Error there",
            "servers": "It cannot say: it would also need the uptime of the servers",
        },
        correct="right",
        why={
            "right": "The contract counts the requests that \"do not fail with Errors\", and an Error is a request \"that returns a 500 error code\". Availability says that the service was there, not that it was right: that needs a measure of its own, such as accuracy.",
            "ninety": "A wrong answer returns no 500 error code, so for the contract it did not fail. See *There when needed* in *The idea*.",
            "servers": "The contract measures availability per request, for each interval of five minutes, not as the uptime of servers. See *There when needed* in *The idea*.",
        },
    )
    choice_uptime
    return (choice_uptime,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Kinds of AI risk](/course/open/rai-kinds-of-ai-risk?lang=en) put reliability issues under malfunctions. This lesson split keeping working into robustness, resilience, availability and predictability, and measured two of them, with the model of [Project: a linear classifier for handwritten digits](/course/open/ml-digits-linear?lang=en). The next lesson, *Risk as chance times harm*, asks how bad a failure is: how likely it is, and how much harm it does.

    **An everyday example.** A bus line can run on time on dry days. Robust is a bus that also runs on time in the rain; resilient, a company that has a spare bus on the road soon after a breakdown; available, a line that runs when people need it; predictable, a bus that leaves at the same minute every day.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_test, choice_uptime, result_noisy, result_pass):
    L.wrap_up(result_noisy, result_pass, choice_test, choice_uptime)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
