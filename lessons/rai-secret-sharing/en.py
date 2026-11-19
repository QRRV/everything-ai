# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-secret-sharing. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-secret-sharing", "en")
    C = L.checks   # five made-up companies: SECRETS, and SHARES, their shares for three parties


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Three short questions about [Training without collecting the data](/course/open/rai-federated-learning?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sends():
    quick_sends = L.choice(
        "quick-sends",
        question="In a round of FedAvg, what does a chosen phone send back to the server?",
        options={
            "data": "Its digits, for the server to train on",
            "right": "Its weights after training",
            "score": "Only how many digits it read right",
        },
        correct="right",
        why={
            "right": "Each chosen phone loads the server's weights, trains on its own data, and sends back only its new weights and its number of examples; the digits stay on the phone.",
            "data": "The digits never leave the phone: that is what federated learning is for. See *The data stay on the phone* in *The idea* of [Training without collecting the data](/course/open/rai-federated-learning?lang=en).",
            "score": "The server needs weights to average, not a score. See *FedAvg: average the weights* in *The idea* of [Training without collecting the data](/course/open/rai-federated-learning?lang=en).",
        },
    )
    quick_sends
    return (quick_sends,)


@app.cell(hide_code=True)
def quick_check_weighted():
    quick_weighted = L.choice(
        "quick-weighted",
        question="Two phones send 0.2 and 0.6 for the same weight. The first trained on 100 examples, the second on 300. What does FedAvg make of them?",
        options={
            "plain": "0.4: the plain mean of the two weights",
            "right": "0.5: each counts by its examples",
            "big": "0.6: the weight of the larger phone",
        },
        correct="right",
        why={
            "right": "Each phone counts by its number of examples: (100 · 0.2 + 300 · 0.6) / 400 = 200 / 400 = 0.5.",
            "plain": "The plain mean counts both phones the same; FedAvg counts each by its number of examples. See *FedAvg: average the weights* in *The idea* of [Training without collecting the data](/course/open/rai-federated-learning?lang=en).",
            "big": "The smaller phone counts too, for a quarter of the examples. See *FedAvg: average the weights* in *The idea* of [Training without collecting the data](/course/open/rai-federated-learning?lang=en).",
        },
    )
    quick_weighted
    return (quick_weighted,)


@app.cell(hide_code=True)
def quick_check_server():
    quick_server = L.choice(
        "quick-server",
        question="After a round of FedAvg, what has the server seen?",
        options={
            "nothing": "Only the average of the weights",
            "right": "The new weights of every chosen phone",
            "data": "The digits of each phone, and its weights",
        },
        correct="right",
        why={
            "right": "Each chosen phone sends its own new weights, and the server averages them itself. Hard et al. write that users must trust the server not to scrutinize those uploads.",
            "nothing": "The server computes the average itself, from the weights each phone sends. See *What the weights still tell* in *The idea* of [Training without collecting the data](/course/open/rai-federated-learning?lang=en).",
            "data": "The digits stay on the phones; only weights travel. See *The data stay on the phone* in *The idea* of [Training without collecting the data](/course/open/rai-federated-learning?lang=en).",
        },
    )
    quick_server
    return (quick_server,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In the previous lesson, the phones kept their digits, yet the server saw the weights of every phone, and users had to trust it not to look at them. Could the server learn only sums, which are all FedAvg's average is made of? The question is older than federated learning: can several parties compute something together without anyone seeing another's input? In Boston, 114 companies did that with their pay data in 2017: the Boston Women's Workforce Council analyzed pay across 166,705 employees while each company's own data stayed private. In this lesson you split numbers into shares that look like random noise, add them up without seeing a single number, and find out why the random numbers must come from the whole range.

    **An everyday example.** Three friends want to know their average salary without saying their own. The first thinks of a large random number, adds their salary, and whispers the result to the second. The second adds their own salary and whispers the result to the third, who adds theirs and whispers it back to the first. The first takes the random number off: what is left is the total of the three salaries, and a third of it is the average. The first two whispers sound like random numbers to whoever hears them; the last one tells the first friend only the total.
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="Shamir (1979) opens with a puzzle from Liu: eleven scientists lock their documents in a cabinet that must open if, and only if, six or more of them are present. With ordinary locks and keys, how many locks does that take at least?",
        options={
            "six": "6: one for each scientist needed",
            "eleven": "11: one for each scientist",
            "right": "462",
        },
        reveal="**462 locks**, with 252 keys for each scientist: \"It is not hard to show that the minimal solution uses 462 locks and 252 keys per scientist.\" Every group of five must lack the key to a lock that any sixth scientist can open, and two groups of five cannot share such a lock, so each of the 462 ways to pick 5 of the 11 needs a lock of its own. Shamir shares a number instead of keys. In this lesson you use the simplest form, in which all shares are needed.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### A secret in shares

    Adi Shamir (1979) answered the puzzle of the cabinet with numbers: "In this paper we show how to divide data D into n pieces in such a way that D is easily reconstructable from any k pieces, but even complete knowledge of k - 1 pieces reveals absolutely no information about D." Shamir calls it a (k, n) threshold scheme. David Evans, Vladimir Kolesnikov and Mike Rosulek (2018) call the pieces shares: "any t − 1 of the shares reveal no information about s, while any t shares allow complete reconstruction of the secret s." Splitting a secret like that is **secret sharing**. Evans et al. often use the case in which "all n shares are necessary and sufficient to reconstruct the secret", and so does this lesson.

    ### Shares that add up

    A simple way adds up: "Another suitable method of sharing is simple additive sharing over a field F", Evans et al. write. With whole numbers it goes like this. Choose a large number M: every share is a whole number from 0 to M − 1. Draw all shares but one at random from that whole range, with `random.randint(0, M - 1)`. The last share is what is left of the secret, modulo M: the secret minus the other shares, then the remainder after dividing by M. In Python that remainder is `%`: `104200 % 100000` is 4200, and for a negative number it is a whole number from 0 to M − 1 too: `-75633 % 100000` is 24367. Think of a clock with M marks instead of 12: adding goes round and round, and only where you end up counts. All shares added up go round the clock and end at the secret.

    One share, or any n − 1 of the n shares, is just random numbers: whatever the secret, every list of values from 0 to M − 1 is equally likely. Evans et al. call additive sharing "secure against n − 1 corrupt parties", and Andrei Lapets and colleagues (2018) describe shares to a general audience as "pieces that on their own are indistinguishable from random noise". That holds only when the random numbers come from the whole range: drawn from a small range, the last share stays close to the secret, as you see after Exercise 1.

    ### Adding without looking

    Shares can be added share by share. Evans et al.: "Additive homomorphism: Given [x] and [y] and a public value z, parties can obtain any of [x + y], [x + z], [xz], without interaction." Here [x] stands for x split into shares. If party 1 holds the first shares of x and y, and party 2 the second shares, each adds up what it holds, and the two totals are shares of x + y. In the same way each of several parties adds up one share of each of many secrets: together their totals give the sum of all secrets, while each party saw only numbers that look like noise. Multiplying two shared numbers is harder: "interaction and communication are required for multiplication gates to compute a sharing [xy] from sharings [x] and [y]."

    Computing a result from the inputs of several parties without anyone seeing the others' inputs is **secure multi-party computation**, MPC for short. The OECD (2023): "MPC is a set of tools that enables the participating parties to jointly compute a function over their input data while keeping those input data private. Essentially, it removes the need for a trusted third party to view and manage the data." It hides the inputs, not what the result itself says: with two parties, the total minus your own number is the other's number. Andrew Yao (1982) asked the classic question: "Two millionaires wish to know who is richer; however, they do not want to find out inadvertently any additional information about each other's wealth. How can they carry out such a conversation?" Yao answered it "by use of one-way functions", not with secret sharing. Since then, Evans et al. write, MPC "has evolved from a theoretical curiosity in the 1980s to a tool for building real systems today."

    ### Who must be trusted

    The uses in Boston and Estonia below assume parties that follow the protocol but look at whatever they get. Evans et al.: "A semi-honest adversary is one who corrupts parties but follows the protocol as specified. In other words, the corrupt parties run the protocol honestly but they may try to learn as much as possible from the messages they receive from other parties." Such parties are "also commonly called honest-but-curious". What it does not cover: if the parties put their shares together, the secrets are out. With additive sharing over n parties, that takes all n of them.

    ### Where it is used

    Evans et al. describe an auction and bidding platform for sugar beet production contracts, built by Danish researchers with the Danish government and stakeholders, as "widely considered to be the first commercial application of MPC". In Boston, Web-MPC, which Lapets et al. describe, used a variant of standard additive secret sharing. The companies sent in their data as shares, Boston University ran the service, and the Boston Women's Workforce Council (BWWC) analyzed the result, in 2015, 2016 and 2017. In 2017 that covered "166,705 employees across 114 companies, comprising roughly 16% of the Greater Boston area workforce". The BWWC's analysis concluded that White women earned 75 cents for every dollar a White man earned; Asian, Black and Hispanic women 71, 52 and 49 cents. Lapets et al. call the honest-but-curious model "appropriate when the service provider and analyzer have no incentives to collude (e.g., if both are vested in an initiative's success)".

    In Estonia, statisticians looked in 2015 for a link between working during university studies and failing to graduate in time. Dan Bogdanov and colleagues (2016) describe how it ran on three servers with additive secret sharing: "Using ten million tax records and half a million education records in the analysis, this is the largest cryptographically private statistical study ever conducted on real data", as they put it. Against anonymizing the data instead, they report that "the use of 3-anonymity caused 10%-30% of sample loss, depending on the demographic group"; 3-anonymity, in which every record shares its identifying values with at least two others, is the k-anonymity of [Hiding in a group: k-anonymity](/course/open/rai-k-anonymity?lang=en) with k = 3. Shares change nothing about the records, only who can see them. The study, Evans et al. write, "found that there was no correlation between working during studies and failure to graduate on time, but that more education was correlated with higher income."

    ### Privacy-enhancing technologies

    The OECD (2023) groups such methods under one name: "Privacy-enhancing technologies (PETs) are a collection of digital technologies and approaches that permit collection, processing, analysis and sharing of information while protecting the confidentiality of personal data." The OECD divides them into four groups, and each **privacy-enhancing technology** falls in one: "data obfuscation, encrypted data processing, federated and distributed analytics and data accountability tools". Federated learning, from the previous lesson, belongs to federated and distributed analytics. MPC belongs to encrypted data processing, beside homomorphic encryption (HE), which computes on encrypted data. Threshold secret sharing, which needs a set number of keys to unlock encrypted data, is an accountability tool for the OECD: "the digital equivalent to a secure box that is locked with multiple separate locks, whose keys are held by different people", like the cabinet of the eleven scientists. Of computing on shares, the OECD writes: "secret sharing techniques or HE can be used to aggregate and compute over data from multiple parties."

    The two can go together. In the previous lesson, McMahan et al. pointed to secure multi-party computation for federated learning. Phones could split their weights, turned into whole numbers, into shares for several servers: the servers would then learn only sums, and FedAvg's average is made of sums.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **secret sharing** | splitting a secret into shares, so that all of them (in Shamir's scheme, any k) give it back and fewer say nothing about it |
        | **share** | one piece of a shared secret: on its own, a number that looks like random noise |
        | **`%`**, modulo M | the remainder after dividing by M, a whole number from 0 to M − 1: `104200 % 100000` is 4200; like a clock with M marks |
        | **secure multi-party computation** (MPC) | methods that let several parties compute a result from their inputs without anyone seeing the others' inputs |
        | **honest-but-curious** | a party that follows the protocol but looks at everything it gets; also called semi-honest |
        | **privacy-enhancing technology** (PET) | a technique to collect, use or share data while protecting the confidentiality of personal data, such as federated learning or MPC |
        """),
        L.watch_out("A share alone says nothing only when the random numbers come from the whole range 0 to M − 1. And modulo M, the shares give back only a number below M: choose M larger than any total you will compute, or the sum goes round the clock and comes out wrong."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To compute a sum of secrets without anyone seeing one, with additive secret sharing:

    1. Choose M larger than any total you will compute. Split each secret: random numbers from 0 to M − 1 for all shares but the last, and as the last share the secret minus those numbers, modulo M. That is `share`.
    2. All shares of a secret, added up modulo M, give it back; fewer say nothing. That is `reveal`.
    3. Give share j of every secret to party j. Each party adds up the shares it holds, modulo M. That is `party_totals`.
    4. The parties put their totals together: `reveal` on the totals gives the sum of all secrets. Only the sum comes out, as long as the parties do not put their shares together.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell shows what `%` does, splits two secrets, x and y, into two shares each, and lets two parties add up the shares they hold.
    """)
    return


@app.cell
def example_shares():
    import random

    M = 100000                     # every share is a whole number from 0 to M - 1
    print(104200 % M, -75633 % M)  # % gives the remainder after dividing by M

    random.seed(1979)
    x = 4200
    x_first = random.randint(0, M - 1)   # a random number from 0 to M - 1
    x_second = (x - x_first) % M         # what is left of x, modulo M
    print("shares of x:", x_first, x_second, "together:", (x_first + x_second) % M)

    y = 1300
    y_first = random.randint(0, M - 1)
    y_second = (y - y_first) % M
    print("shares of y:", y_first, y_second, "together:", (y_first + y_second) % M)

    # party 1 holds both first shares, party 2 both second shares; each adds up what it holds
    total_1 = (x_first + y_first) % M
    total_2 = (x_second + y_second) % M
    print("party totals:", total_1, total_2, "together:", (total_1 + total_2) % M)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `104200 % M` is `4200`, and `-75633 % M` is `24367`. The shares of x, `11351` and `92849`, look nothing like 4200, yet they add up to `104200`, which modulo M is `4200`; the same goes for y. Party 1 holds `11351` and `69401`, party 2 holds `92849` and `31899`: numbers that say nothing on their own. Their totals, `80752` and `24748`, add up to `105500`, which modulo M is `5500`: the sum of x and y, though neither party saw x or y. In Exercise 1 you split a secret into any number of shares.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_share():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. share: split a secret

        `share(secret, randoms, M)` gets a whole number `secret` from 0 to M − 1, a list `randoms` of whole numbers from 0 to M − 1, and the number `M`. It gives back the list `shares`: the random numbers, in their order, and after them the last share, what is left of `secret` once all of `randoms` are taken off, modulo `M`. Inside, `left` starts at `secret`, and the loop puts each random number `r` in `shares`. Replace the two `...`. The check calls `share(4200, [71320, 8513], 100000)`, then other secrets and random numbers.
        """),
        L.task("Give back the random numbers, then the share that makes them all add up to the secret, modulo M."),
    ])
    return


@app.cell(hide_code=True)
def exercise_share():
    box_share = L.code_box("share", lines=9)
    box_share
    return (box_share,)


@app.cell(hide_code=True)
def check_share(box_share):
    result_share = L.check("share", box_share.value)
    result_share
    return (result_share,)


@app.cell(hide_code=True)
def hints_share():
    L.hints(
        "share",
        [
            "Step 1 of the recipe. In the worked example, `x_second` is what is left of `x` once `x_first` is taken off, modulo `M`.",
            "Here there can be any number of random numbers. The loop already puts each one in `shares`; take it off `left` as well, so that after the loop `left` is the secret minus all of them. The last share must be a whole number from 0 to M − 1, like the others.",
            "`left = left - r` and `shares.append(___)`",
        ],
        why="`left = left - r` takes each random number off, and `shares.append(left % M)` adds the last share after the loop. `share(4200, [71320, 8513], 100000)` gives `[71320, 8513, 24367]`: together `104200`, and `104200 % 100000` is `4200`.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What one share gives away

    The picture shares two secrets, 20,000 and 70,000, 200 times each into two shares, with your `share` once Exercise 1 is right and the course's until then. Each time, the random number, the first share, is drawn from 0 to W − 1. Each dot is the last share of one sharing: filled for 20,000, open for 70,000, with a dashed line at the secret. The first shares need no picture: they are random numbers from 0 to W − 1 whatever the secret. With the slider you choose W.
    """)
    return


@app.cell(hide_code=True)
def demo_leak(box_share, result_share):
    _yours = C.mine(box_share.value, result_share.state == "correct")
    _runs = C.leak_runs(_yours) if _yours else None
    _who = "your `share`"
    if _runs is None:
        _who = "the course's `share`, as yours stops with an error here" if _yours else "the course's `share`, as Exercise 1 is not right yet"
        _runs = C.leak_runs()
    _out = mo.vstack([
        mo.md(f"Made with {_who}."),
        L.explore("leak", "W", lambda _w: C.leak_frame(_runs, int(_w), L.lang), start=1000, values=C.WIDTHS, role="",
                  task="Drag W from 1,000 up to 100,000: from which W on can a last share no longer tell the two secrets apart?"),
    ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_reveal():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. reveal: put a secret back together

        `reveal(shares, M)` gets a list `shares` of whole numbers and the number `M`. It gives back the shares added up, modulo `M`. Inside, `total` starts at 0, and the loop goes over the shares, one share `s` at a time. Replace the two `...`. The check calls `reveal([71320, 8513, 24367], 100000)`, then other shares.
        """),
        L.task("Give back the shares added up, modulo M."),
    ])
    return


@app.cell(hide_code=True)
def exercise_reveal():
    box_reveal = L.code_box("reveal", lines=6)
    box_reveal
    return (box_reveal,)


@app.cell(hide_code=True)
def check_reveal(box_reveal):
    result_reveal = L.check("reveal", box_reveal.value)
    result_reveal
    return (result_reveal,)


@app.cell(hide_code=True)
def hints_reveal():
    L.hints(
        "reveal",
        [
            "Step 2 of the recipe. In the worked example, `(x_first + x_second) % M` puts x back together from its two shares.",
            "Here there can be any number of shares: let the loop add each one to `total`, so that after the loop `total` holds all of them added up. That sum may be M or more.",
            "`total = ___` and `return total % M`",
        ],
        why="`total = total + s` adds the shares up, and `return total % M` takes the remainder. `reveal([71320, 8513, 24367], 100000)` gives `4200`; without the last share, `reveal([71320, 8513], 100000)` gives `79833`, nothing like it.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_party_totals():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. party_totals: what each party adds up

        `SECRETS` holds what five made-up companies pay their women employees in a year, in thousands of dollars: `[3120, 870, 12450, 5300, 760]`. Each company split its number into three shares with the course's `share`, one for each of three parties. `SHARES` holds them: `SHARES[i][j]` is share j of company i, and party j holds it. Counting starts at 0, so the parties are 0, 1 and 2.

        `party_totals(table, M)` gets such a table, a list `table` with one row per company and in each row one share per party, and the number `M`. It gives back the list `totals`: for each party j, the shares it holds added up, modulo `M`. Inside, the loop over `j` goes over the parties; for each, `total` starts at 0, and the loop over the rows takes one row `row` at a time. Replace the two `...`. The check calls `party_totals(SHARES, 100000)`, then other tables.
        """),
        L.task("Give back, for each party, the shares it holds added up, modulo M."),
    ])
    return


@app.cell(hide_code=True)
def exercise_party_totals():
    box_party_totals = L.code_box("party-totals", lines=9)
    box_party_totals
    return (box_party_totals,)


@app.cell(hide_code=True)
def check_party_totals(box_party_totals):
    result_party_totals = L.check("party-totals", box_party_totals.value)
    result_party_totals
    return (result_party_totals,)


@app.cell(hide_code=True)
def hints_party_totals():
    L.hints(
        "party-totals",
        [
            "Step 3 of the recipe. In the worked example, party 1 adds up `x_first` and `y_first`: the first share of each secret.",
            "Party j holds column j: from each company's row, the share at place j. The inner loop goes over the rows; add that share of each row to `total`. After that loop, `totals` gets the remainder of `total` after dividing by `M`.",
            "`total = total + ___` and `totals.append(total % M)`",
        ],
        why="`total = total + row[j]` adds up column j, and `totals.append(total % M)` keeps the total as a share. `party_totals(SHARES, 100000)` gives `[38021, 53499, 30980]`; added up, modulo 100,000, that is `22500`, the sum of the five secrets.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_two_companies():
    mo.md(r"""
    ### Exercise 4. Two companies

    Only two companies take part in a sharing like the one of Exercise 3, and the parties publish the total of their two numbers.
    """)
    return


@app.cell(hide_code=True)
def choice_two_companies_cell():
    choice_two_companies = L.choice(
        "two-companies",
        question="What can each company work out from the total?",
        options={
            "right": "The other's number: the total minus its own",
            "nothing": "Nothing: the shares hid every number",
            "half": "Only that the other's is about half the total",
        },
        correct="right",
        why={
            "right": "Secure multi-party computation hides the inputs, not what the result itself says: with two companies, the total minus your own number is the other's number. With 114 companies, as in Boston in 2017, the total minus your own number is the sum of the other 113.",
            "nothing": "The shares hid the numbers on the way, but the total is the result, and the result says what it says: the total minus your own number is the other's. See *Adding without looking* in *The idea*.",
            "half": "The total minus your own number is exactly the other's number, not an estimate. See *Adding without looking* in *The idea*.",
        },
    )
    choice_two_companies
    return (choice_two_companies,)


@app.cell(hide_code=True)
def text_exercise_collude():
    mo.md(r"""
    ### Exercise 5. Two parties together

    Suppose Boston had used exactly the scheme of this lesson, with two parties: each company splits its number into two shares, one for Boston University and one for the BWWC.
    """)
    return


@app.cell(hide_code=True)
def choice_collude_cell():
    choice_collude = L.choice(
        "collude",
        question="What could the two learn if they put together what they hold?",
        options={
            "total": "Still only the totals",
            "right": "Each company's own numbers",
            "noise": "Only noise, since each share looks like noise",
        },
        correct="right",
        why={
            "right": "Together the two hold both shares of every company, and both shares of a number, added up modulo M, give it back. The honest-but-curious model trusts them not to do that; Lapets et al. call it appropriate when the two have no incentives to collude.",
            "total": "The totals are what they compute on purpose, but together they also hold both shares of each company, and those give each company's number back. See *Who must be trusted* in *The idea*.",
            "noise": "One share alone looks like noise, but all shares of a number together give it back, and the two together hold all of them. See *Who must be trusted* in *The idea*.",
        },
    )
    choice_collude
    return (choice_collude,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Training without collecting the data](/course/open/rai-federated-learning?lang=en), the phones kept their data, but the server saw the weights of every phone. Here several parties compute a sum without anyone seeing a single input: shares that look like noise, added up share by share, give back only the total. Like federated learning, it is a privacy-enhancing technology, and the two can go together. In the next lesson, *What a model remembers*, the question turns to the model itself: what a trained model can give away about the data it learned from.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_collude, choice_two_companies, result_party_totals, result_reveal, result_share):
    L.wrap_up(result_share, result_reveal, result_party_totals, choice_two_companies, choice_collude)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
