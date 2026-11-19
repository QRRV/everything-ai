# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-ml-threat-models. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    from everything_ai.lesson import lesson

    L = lesson("rai-ml-threat-models", "en")
    C = L.checks   # a small model the course trains itself: stop signs and speed limit signs of 8 by 8 pixels


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions on [Thinking like an attacker](/course/open/rai-security-basics?lang=en) and [The machine learning workflow](/course/open/ml-workflow?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_cia():
    quick_cia = L.choice(
        "quick-cia",
        question="Someone breaks into the database of a navigation app and raises the speed limits of a few roads. Which property does that break?",
        options={
            "right": "Integrity: someone changes data they may not change",
            "conf": "Confidentiality: someone reads what they may not read",
            "avail": "Availability: drivers can no longer use the app",
        },
        correct="right",
        why={
            "right": "Changing what you have no right to change is a loss of integrity. That the intruder could get in at all is a security problem too, but what goes wrong for the drivers is that the data is no longer right.",
            "conf": "The speed limits are not secret. What goes wrong is that they are changed. See *Three things to protect* in [Thinking like an attacker](/course/open/rai-security-basics?lang=en).",
            "avail": "The app keeps working for everyone, with the wrong limits. See *Three things to protect* in [Thinking like an attacker](/course/open/rai-security-basics?lang=en).",
        },
    )
    quick_cia
    return (quick_cia,)


@app.cell(hide_code=True)
def quick_check_workflow():
    quick_workflow = L.choice(
        "quick-workflow",
        question="In the machine learning workflow, when do you measure the model on the test set?",
        options={
            "right": "Once, at the end, when everything is chosen",
            "round": "In every round, to choose what to change next",
            "start": "At the start, to set the goal",
        },
        correct="right",
        why={
            "right": "The test set waits until the end: choose with it, and the score bends towards it.",
            "round": "Choosing with the test set makes its score too rosy; the rounds use the validation set. See *Recipe* in [The machine learning workflow](/course/open/ml-workflow?lang=en).",
            "start": "The goal is set before you see any results, and without the test set. See *Recipe* in [The machine learning workflow](/course/open/ml-workflow?lang=en).",
        },
    )
    quick_workflow
    return (quick_workflow,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    So far in this module, attacks went through what surrounds a program: its accounts, its database, its code. A model can be attacked too, through what it is made for: answering. An attacker can change an input a little, so that the model gets it wrong, or ask question after question, until its answers give away what it learned from. This lesson sorts those attacks, the way the US standards institute NIST does, and then makes one: you will turn a stop sign into a speed limit sign, for a small model, first with its weights and then without them.

    **An everyday example.** A spam mail writes *V1agra* instead of *Viagra*. A reader sees the same word; a filter that looks for the word sees another one. The mail changed a little, and only for the filter.
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="Goodfellow and colleagues showed GoogLeNet, a neural network, a photo of a panda. That is a far bigger model than the ones so far, of a kind that later lessons build. Panda, said the network, with 57.7% confidence. Then they moved every pixel up or down by the smallest step of an 8-bit image, in a direction they worked out from the network. What did the network say about the new photo?",
        options={
            "same": "Panda, with a little less confidence",
            "right": "Gibbon, with 99.3% confidence",
            "unsure": "Unsure: no class above 50%",
        },
        reveal="**Gibbon, with 99.3% confidence.** The authors: \"Here our ε of .007 corresponds to the magnitude of the smallest bit of an 8 bit image encoding after GoogLeNet’s conversion to real numbers.\" Shown the pattern of signs on its own, the network said nematode, with 8.2% confidence. A year earlier, Szegedy and colleagues had named such inputs: \"We term the so perturbed examples 'adversarial examples'.\"",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### What attackers want from a model

    NIST's report *Adversarial Machine Learning* (NIST AI 100-2 E2025, March 2025) sorts attacks on machine learning along four axes: "1) learning method and stage of the learning process when the attack is mounted, 2) attacker goals and objectives, 3) attacker capabilities, and 4) attacker knowledge of the learning process." For models that predict, such as a classifier, it names three goals: "availability breakdown, integrity violation, and privacy compromise". They are the three properties of [Thinking like an attacker](/course/open/rai-security-basics?lang=en), with privacy in the place of confidentiality: "While the term 'confidentiality' is more widely used in taxonomies of traditional cybersecurity attacks, the AML field has tended to use the top-level term 'privacy'". AML is adversarial machine learning, the study of attacks on machine learning and defenses against them.

    - **Availability breakdown**: the attacker disrupts "the ability of other users or processes to obtain timely and reliable access to its services."
    - **Integrity violation**: the attacker forces the model "to misperform against its intended objectives and produce predictions that align with the adversary’s objective."
    - **Privacy compromise**: "the unintended leakage of restricted or proprietary information from a PredAI system, including details about a model’s training data, weights, or architecture". PredAI is NIST's word for predictive AI; the architecture is how the model is built.

    For generative AI, NIST adds a fourth goal, misuse enablement, in which "an attacker seeks to deliberately circumvent technical restrictions" that the owner of the system imposes on its use. *Jailbreaks*, later in this module, is about that one.

    ### When: while it learns, or while it is used

    The stage tells two families apart. A poisoning attack strikes while a model learns: the attacker slips something into its training data. An **evasion attack** strikes when the model is used: the attacker changes an input. NIST files evasion under integrity. Poisoning can serve two of the goals: data poisoning that spoils the model as a whole is filed under availability, and targeted and backdoor poisoning, which make the model wrong for inputs the attacker chooses, under integrity. *Poisoning the training data*, later in this module, is about the first family; this lesson is about the second.

    ### What the attacker knows

    NIST: "**White-box** attacks. These assume that the attacker operates with full knowledge about the ML system, including the training data, model architecture, and model hyperparameters." "**Black-box** attacks. These attacks assume that the attacker operates with minimal, and sometimes no knowledge at all about the ML system." In between lie gray-box attacks. Black box does not mean powerless; NIST calls those attacks "the most practical since they assume that the attacker has no knowledge of the AI system and utilizes system interfaces readily available for normal use": anyone who can ask the model a question.

    ### Adversarial examples

    NIST: "In an evasion attack, the adversary’s goal is to generate adversarial examples: samples whose classification can be changed to an arbitrary class of the attacker’s choice – often with only minimal perturbation". The idea is older than deep learning: NIST dates early known evasion attacks back to 1988, and adversarial examples for linear classifiers in spam filters to 2004. Szegedy and colleagues (2014) found that for neural networks, "applying an imperceptible non-random perturbation to a test image, it is possible to arbitrarily change the network’s prediction".

    ### The fast gradient sign method

    Goodfellow, Shlens and Szegedy (2015) gave a quick way to make one: move every input value by the same small amount ε, in the direction that raises the loss of the right answer. In their notation, the change is η = ε · sign(∇ₓJ(θ, x, y)): ∇ₓJ is the gradient of the loss J with respect to the input x, not the parameters θ, y is the right answer, and sign keeps only the sign of each slope, +1 or −1. "We refer to this as the 'fast gradient sign method' of generating adversarial examples." For a logistic regression it is simple. Making pixel j one unit higher changes the logit by its weight wⱼ, and for a stop sign a higher logit means a higher loss. So the direction of pixel j is the sign of wⱼ, and the logit rises by ε times the sum of the sizes of all weights, ε · Σ|wⱼ|. For neural networks, Goodfellow and colleagues name this behavior as the cause: "the primary cause of neural networks’ vulnerability to adversarial perturbation is their linear nature."

    ### Outside the computer

    Eykholt and colleagues (2018) attacked a real stop sign "with a perturbation in the form of only black and white stickers". The model they attacked read it as a speed limit sign in "100% of the images obtained in lab settings, and in 84.8% of the captured video frames obtained on a moving vehicle". They knew that model through and through: "We focus on the white-box setting as well". Real attackers can get by with much less. At a detector of phishing pages, NIST reports, researchers found 100 adversarial examples among 4600 samples that the detector was unsure about, made with "relatively simple methods for evasion, such as image cropping, masking, or blurring techniques." MITRE keeps a public catalog of such attacks, ATLAS.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **evasion attack** | an attack while a model is used: a changed input makes it give the wrong answer |
        | **adversarial example** | an input changed on purpose, often by very little, so that a model gets it wrong |
        | **white box** | the attacker knows everything about the model: weights, architecture, training data |
        | **black box** | the attacker knows little or nothing about the model, and can usually only ask it questions |
        | **FGSM** | the fast gradient sign method: every input value moves by ε, in the direction of the sign of the gradient |
        | **ε** | here: how far FGSM moves every input value; `eps` in code |
        | `np.sign(a)` | the sign of every number in the array `a`: 1, −1, or 0 for 0 |
        """),
        L.watch_out("Black box does not mean that the attacker can do nothing: NIST calls black-box attacks the most practical. And the stickers did not fool the model every time: 100% was in the lab; from a moving car, 84.8% of the video frames, for one model the researchers knew completely."),
        mo.Html(C.idea_picture(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To fool a classifier with a small change to an input, along its gradient:

    1. Choose the input and the answer you want the model to give.
    2. Find, for every input value, the direction that moves the model towards that answer: the sign of the gradient of the loss of the right answer. For our logistic regression, where the answer we want is a speed limit sign (a higher logit), that is the sign of each weight.
    3. Move every input value by the same small ε in its direction.
    4. Work out the smallest ε that turns the answer, and check that the model now gives the answer you wanted.
    5. Without the weights, find the directions with questions: change one input value a little, and see which way the answer goes.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The course trained a logistic regression on made-up pictures of 8 by 8 pixels, each pixel a number between 0 and 1 (in the pictures, the higher the number, the more its cell stands out from the page): 100 stop signs and 100 speed limit signs. It has 64 weights `C.W`, one for each pixel, and a bias term `C.B`; its probability is that of a speed limit sign. On 200 new pictures, it makes no mistakes. `C.STOP` is a stop sign it has not seen. The cell moves every pixel of `C.STOP` by 0.1, once up or down at random (`C.RANDOM_SIGNS` holds 64 times 1 or −1, drawn at random) and once along the signs of the weights, and works out the logit and the probability each time.
    """)
    return


@app.cell
def example_steps():
    # the course's model: 64 weights and a bias term, for pictures of 8 by 8 pixels
    w, b = C.W, C.B
    z = np.dot(w, C.STOP) + b
    print("STOP: logit", round(z, 3), "P(speed limit)", round(1 / (1 + np.exp(-z)), 3))
    # every pixel 0.1 up or down: at random, or along the signs of the weights
    for name, step in [("random", 0.1 * C.RANDOM_SIGNS), ("along the signs", 0.1 * np.sign(w))]:
        z = np.dot(w, C.STOP + step) + b
        print(name + ": logit", round(z, 3), "P(speed limit)", round(1 / (1 + np.exp(-z)), 3))
    print("sum of the sizes of the weights:", round(np.sum(abs(w)), 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    STOP has a logit of −3.845: a speed limit sign with probability 0.021. At random, the steps of 0.1 mostly cancel each other out; here the logit even goes down a little, to −4.209. Along the signs, every step pushes the logit up: by 0.1 times the size of its weight, together 0.1 × 18.18 = 1.818, to −2.027. Still a stop sign, with probability 0.116 for a speed limit sign. Steps of 0.1 are not enough.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_fgsm():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. fgsm: one step along the signs

        Write `fgsm(image, w, eps)`: the picture `image` with every pixel moved by `eps`, up where its weight in `w` is positive and down where it is negative. Replace the `...`. The check calls `fgsm(STOP, W, 0.1)` with the picture and the weights of the worked example, then other pictures, other steps and a second model. fgsm may push a pixel below 0 or above 1; a real attack would also keep every pixel between 0 and 1, and the check accepts both.
        """),
        L.task("Move every pixel of image by eps, in the direction of the sign of its weight."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fgsm():
    box_fgsm = L.code_box("fgsm", lines=3)
    box_fgsm
    return (box_fgsm,)


@app.cell(hide_code=True)
def check_fgsm(box_fgsm):
    result_fgsm = L.check("fgsm", box_fgsm.value)
    result_fgsm
    return (result_fgsm,)


@app.cell(hide_code=True)
def hints_fgsm():
    L.hints(
        "fgsm",
        [
            "Step 3 of the recipe, as in the last lines of the worked example.",
            "Take the signs of the weights with `np.sign`, multiply them by `eps`, and add the result to the picture.",
            "`return image + eps * ___`",
        ],
        why="`return image + eps * np.sign(w)`. `fgsm(STOP, W, 0.1)` moves every pixel by 0.1 along the signs: the model still reads a stop sign, with probability 0.116 for a speed limit sign.",
    )
    return


@app.cell(hide_code=True)
def text_after_fgsm():
    mo.md(r"""
    With eps = 0.25, the model reads a speed limit sign, with probability 0.668 (0.623 if your fgsm keeps every pixel between 0 and 1). Your fgsm never asked what the picture shows: the direction comes from the model alone, and is the same for every stop sign.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_eps():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. smallest_eps: how far, exactly?

        The model reads a speed limit sign once the logit is above 0: the decision boundary of [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en). One step of fgsm with eps raises the logit by eps times the sum of the sizes of the weights. Write `smallest_eps(image, w, b)`: the eps at which the logit of `fgsm(image, w, eps)` is exactly 0. `abs` works on an array too: it gives the size of every number in it. Replace the `...`. The check calls `smallest_eps(STOP, W, B)`, then another picture and the second model.
        """),
        L.task("Give back the eps that brings the logit of the model exactly to 0."),
    ])
    return


@app.cell(hide_code=True)
def exercise_eps():
    box_eps = L.code_box("smallest-eps", lines=4)
    box_eps
    return (box_eps,)


@app.cell(hide_code=True)
def check_eps(box_eps):
    result_eps = L.check("smallest-eps", box_eps.value)
    result_eps
    return (result_eps,)


@app.cell(hide_code=True)
def hints_eps():
    L.hints(
        "smallest-eps",
        [
            "Step 4 of the recipe. In the worked example, steps of 0.1 raised the logit by 0.1 times the sum of the sizes of the weights.",
            "Work out the logit `z` of `image` with `np.dot`, `w` and `b`. A step with eps raises it by eps times the sum of the sizes of the weights. Which eps brings `z` exactly to 0?",
            "`z = np.dot(w, image) + b`, then `return -z / ___`",
        ],
        why="`z = np.dot(w, image) + b` and `return -z / np.sum(abs(w))`. For STOP: 3.845 / 18.18, so 0.212 per pixel.",
    )
    return


@app.cell(hide_code=True)
def text_after_eps():
    mo.md(r"""
    STOP needs 0.212 per pixel, and the other 99 stop signs of the test set need between 0.172 and 0.257. On a scale from 0 to 1, that is a lot: in the picture in *The idea*, at 0.22, you can see the change. The panda photo needed only the smallest step of an 8-bit image. Our model adds up only 64 pixels; a photo for GoogLeNet has far more input values, so far more small steps can add up. Goodfellow and colleagues saw the main cause of this weakness of neural networks in their linear nature.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Along the signs, or at random

    Drag eps from 0 to 0.3. The picture moves every pixel of STOP by eps twice: along the signs of the weights, as your fgsm does, and up or down at random.
    """)
    return


@app.cell(hide_code=True)
def try_eps_cell():
    try_eps = mo.ui.slider(start=0, stop=0.3, step=0.01, value=0.15, label="eps")
    try_eps
    return (try_eps,)


@app.cell(hide_code=True)
def demo_eps(try_eps):
    _out = mo.Html(C.demo_picture(try_eps.value, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Along the signs, the model turns between 0.21 and 0.22. At random, it does not turn even at 0.3: it is even surer of the stop sign there, with probability 0.007 for a speed limit sign. The same amount of change, in another direction, does nothing for the attacker.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_signs():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. guess_signs: the same direction, without the weights

        So far you knew the weights: a white box. Behind a website, an attacker sees only the answers: a black box. `ask(picture)` answers as such a model does: the probability of a speed limit sign, nothing else. Write `guess_signs(ask, image, delta)`: ask once about `image`; then, for each pixel, about a copy of `image` in which only that pixel is `delta` higher. Keep 1 when the answer went up and −1 when not, and give back the 64 signs with `np.array`. A copy of the picture is `np.array(image)`. Replace the `...`. The check calls `guess_signs(ask, STOP, 0.01)`, then another picture and a second model.
        """),
        L.task("Find the sign of every weight from the answers of ask alone."),
    ])
    return


@app.cell(hide_code=True)
def exercise_signs():
    box_signs = L.code_box("signs", lines=12)
    box_signs
    return (box_signs,)


@app.cell(hide_code=True)
def check_signs(box_signs):
    result_signs = L.check("signs", box_signs.value)
    result_signs
    return (result_signs,)


@app.cell(hide_code=True)
def hints_signs():
    L.hints(
        "signs",
        [
            "Step 5 of the recipe: one question for the picture, then one for each pixel, with a loop as in [Repeating: for loops](/course/open/py-for?lang=en).",
            "Keep the answer for `image` itself. Then, for each `i` in `range(len(image))`, make a copy with `np.array(image)`, add `delta` to its pixel `i`, ask about the copy, and append 1 or −1 to a list.",
            "`if ask(bumped) > ___:`, with `bumped = np.array(image)` and `bumped[i] = bumped[i] + delta` above it, in the loop",
        ],
        why="`base = ask(image)` and `signs = []`; then, for each `i` in `range(len(image))`: `bumped = np.array(image)`, `bumped[i] = bumped[i] + delta`, and `signs.append(1)` if `ask(bumped) > base`, else `signs.append(-1)`; at the end, `return np.array(signs)`. For STOP, all 64 signs are right after 65 questions.",
    )
    return


@app.cell(hide_code=True)
def text_after_signs():
    mo.md(r"""
    With 65 questions, the attacker has the same direction as with the weights, and fgsm works as before. Our `ask` gives a probability, so even a small change shows in its answer. NIST mentions black-box attacks that are efficient with their questions, needing "typically less than 1000". An attacker who cannot ask question after question still has a way: Szegedy and colleagues found that "the same perturbation can cause a different network, that was trained on a different subset of the dataset, to misclassify the same input." So an attacker can make the change on a model of their own, and try it on the real one.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_goal():
    mo.md(r"""
    ### Exercise 4. Which goal?

    An attacker sends a model of a hospital thousands of questions, and works out from the answers which patients were in its training data.
    """)
    return


@app.cell(hide_code=True)
def choice_goal_cell():
    choice_goal = L.choice(
        "goal",
        question="Which of NIST's goals does the attacker have?",
        options={
            "right": "Privacy compromise: data leaks out of the model",
            "integ": "Integrity violation: wrong answers",
            "avail": "Availability breakdown: others can no longer use it",
        },
        correct="right",
        why={
            "right": "NIST: \"A PRIVACY COMPROMISE attack causes the unintended leakage of restricted or proprietary information from a PredAI system, including details about a model’s training data, weights, or architecture\". In the words of [Thinking like an attacker](/course/open/rai-security-basics?lang=en), it is a loss of confidentiality.",
            "integ": "The answers may all be right: the attacker only reads them, and learns from them. An integrity violation forces the model to give predictions that suit the attacker. See *What attackers want from a model* in *The idea*.",
            "avail": "Everyone can still use the model. What goes wrong is what the answers give away. See *What attackers want from a model* in *The idea*.",
        },
    )
    choice_goal
    return (choice_goal,)


@app.cell(hide_code=True)
def text_exercise_knowledge():
    mo.md(r"""
    ### Exercise 5. What did they know?

    Eykholt and colleagues made black and white stickers for a real stop sign. In the lab, the model they attacked read every photo of it as a speed limit sign.
    """)
    return


@app.cell(hide_code=True)
def choice_knowledge_cell():
    choice_knowledge = L.choice(
        "knowledge",
        question="What did they know about that model?",
        options={
            "right": "Everything, weights included: white box",
            "black": "Only its answers to their photos: black box",
            "none": "Nothing at all: stickers fool every model",
        },
        correct="right",
        why={
            "right": "The paper: \"We focus on the white-box setting as well\". With the model in hand, they could follow its gradient, as your fgsm follows the signs of the weights.",
            "black": "They had more than answers: \"We focus on the white-box setting as well\", the paper says. A black-box attack works too, as in Exercise 3, but this was not one. See *What the attacker knows* in *The idea*.",
            "none": "The stickers were made for one model, and the paper says: \"We focus on the white-box setting as well\". From a moving car, they worked in 84.8% of the video frames for that model. See *Outside the computer* in *The idea*.",
        },
    )
    choice_knowledge
    return (choice_knowledge,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Thinking like an attacker](/course/open/rai-security-basics?lang=en), security protected confidentiality, integrity and availability; NIST sorts attacks on models by the same three goals, with privacy for confidentiality. In [The machine learning workflow](/course/open/ml-workflow?lang=en), the test set measured the model once, at the end. Our model gets all 200 test pictures right, and STOP still turns with 0.212 per pixel: a test set measures the model on inputs like the ones it learned from, not on inputs made to fool it. In [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en), the model said yes on one side of a line; `smallest_eps` is how far a picture is from that line, in steps along the signs. *The OWASP lists for LLM apps* turns to applications built on language models.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_goal, choice_knowledge, result_eps, result_fgsm, result_signs):
    L.wrap_up(result_fgsm, result_eps, result_signs, choice_goal, choice_knowledge)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
