# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-secrets-and-endpoints. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-secrets-and-endpoints", "en")
    C = L.checks   # the shop's made-up repository with two keys in its history, addresses, tokens, the course's server


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions on [Injection and secrets](/course/open/rai-classic-vulnerabilities?lang=en) and [Batches, keys and the bill](/course/open/build-batch-keys-costs?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_uber():
    quick_uber = L.choice(
        "quick-uber",
        question="According to a complaint by the US Federal Trade Commission, where did intruders find the access key into Uber's files in 2016?",
        options={
            "right": "In code in a private repository on GitHub",
            "public": "In a public repository on GitHub",
            "laptop": "On a laptop that an employee left on a train",
        },
        correct="right",
        why={
            "right": "The complaint says the key was \"in plain text in code that was posted to a private GitHub repository\". Whoever got into the repository had the key.",
            "public": "The repository was private; the intruders said they got in with passwords exposed in earlier breaches. See *Secrets in code* in [Injection and secrets](/course/open/rai-classic-vulnerabilities?lang=en).",
            "laptop": "The key was in code, in a private repository on GitHub. See *Secrets in code* in [Injection and secrets](/course/open/rai-classic-vulnerabilities?lang=en).",
        },
    )
    quick_uber
    return (quick_uber,)


@app.cell(hide_code=True)
def quick_check_environ():
    quick_environ = L.choice(
        "quick-environ",
        question="Your code reads `os.environ[\"MODEL_API_KEY\"]`, and that environment variable is not set. What happens?",
        options={
            "right": "Python stops with a KeyError",
            "empty": "The key is the empty text",
            "none": "The value is None, and the request goes without a key",
        },
        correct="right",
        why={
            "right": "`os.environ` holds the environment variables as a dictionary, and looking up a name that is not in a dictionary gives a KeyError.",
            "empty": "With square brackets, a name that is missing is a KeyError, as in every dictionary. See *New words and symbols* in [Batches, keys and the bill](/course/open/build-batch-keys-costs?lang=en).",
            "none": "With square brackets, a name that is missing is a KeyError, as in every dictionary; the request is never sent. See *New words and symbols* in [Batches, keys and the bill](/course/open/build-batch-keys-costs?lang=en).",
        },
    )
    quick_environ
    return (quick_environ,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A key in code is a key for everyone who can read the code, and a server is a door for everyone who can reach it. Both happen a lot. Michael Meli, Matthew McNiece and Bradley Reaves scanned public GitHub in 2017 and 2018, and looked only for private key files and for the keys of 11 services whose keys have a form of their own. They found 201,642 different secrets, in over 100,000 repositories; about two weeks after they found them, 81% were still there. The security company GitGuardian counted 28.65 million new secrets in public commits on GitHub in 2025, 34% more than in 2024; the secrets for AI services it found rose by 81%, to 1,275,105. And researchers at Cisco looked in 2025 for servers of Ollama, a program that runs LLMs, that could be reached from the internet: they found 1,139, and 214 of them answered with a model.

    **An everyday example.** A spare key under the doormat. Anyone who knows where to look can come in, and the people you would least like to come in know where to look.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Meli, McNiece and Reaves put a text of their own in a public commit on GitHub, again and again, and timed how soon GitHub's search showed it. What was the median time?",
        options={
            "right": "20 seconds",
            "minutes": "15 minutes",
            "day": "A whole day",
        },
        reveal="**20 seconds.** In their words: \"We found that the median time to discovery was 20 seconds, with times ranging from half a second to over 4 minutes, and no discernible impact from time-of-day.\" That is how soon someone who watches GitHub's search could see a new key; they timed no attacker.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### The attack surface

    NIST defines the **attack surface** as "The set of points on the boundary of a system, a system component, or an environment where an attacker can try to enter, cause an effect on, or extract data from, that system, component, or environment." For the help assistant of Spoke & Chain, the lessons before looked at one point: the text the model reads. This lesson looks at two more: the keys in its code, and the server it runs on. OWASP's cheat sheet on secrets: "Consider secrets as part of the attack surface during threat modeling exercises." A point you take away is one less to defend: a key that is not in the code cannot leak from the code.

    ### Keys out of the code

    A key in the code goes wherever the code goes. The Twelve-Factor App, a set of rules for apps by Adam Wiggins, puts it as a test: "A litmus test for whether an app has all config correctly factored out of the code is whether the codebase could be made open source at any moment, without compromising any credentials." Its rule: "The twelve-factor app stores config in environment variables", as [Batches, keys and the bill](/course/open/build-batch-keys-costs?lang=en) did with `os.environ`. Not everyone finds that enough. In its part on containers, OWASP's cheat sheet on secrets warns that "environment variables are generally accessible to all processes and may be included in logs or system dumps", and does not recommend them "unless the other methods are not possible". Both agree on one thing: not in the code.

    ### Secret scanning

    **Secret scanning** is searching code and its history for texts that have the form of a key, a password or a token. Many services give their keys a form of their own, such as a fixed start and a fixed length, and that form is what a scanner looks for. Meli and colleagues did the same: they searched for private key files and for "11 high-impact platforms with distinctive API key formats". GitHub's secret scanning "scans your entire Git history on all branches of your repository for hardcoded credentials". It runs by itself only on public repositories; private repositories of an organization need GitHub Secret Protection, on a GitHub Team or Enterprise Cloud plan. Push protection comes one step earlier: it "blocks pushes that contain secrets before they reach your repository". For your own account it is on by default, but then it stops only pushes to public repositories. For a repository it is off by default and needs GitHub Secret Protection, and by default anyone with write access to the repository can bypass push protection with a reason, such as *I'll fix it later*.

    A commit that removes a key does not remove the commit that added it: the key stays in the history. So GitHub's first advice for a key a scanner found is the one from [Injection and secrets](/course/open/rai-classic-vulnerabilities?lang=en): "rotate the affected credential immediately to prevent unauthorized access". Meli and colleagues even got back commits that had been removed by rewriting the history, on GitHub as it worked in 2018.

    ### Open doors

    A server listens on an address and a port, and the address decides who can reach it. Addresses that start with 127 are, in RFC 1122, the rules for internet hosts of 1989, an "Internal host loopback address", and "Addresses of this form MUST NOT appear outside a host." A server on 127.0.0.1 can be reached only from the computer it runs on, and `localhost` is the name of that address. For a server, the address 0.0.0.0, or the empty text in Python, means the opposite: every address of the computer, also the one the network reaches it on. Python's own `python -m http.server` does that: "By default, the server binds itself to all interfaces."

    Ollama listens on 127.0.0.1, port 11434, by default; its FAQ shows how the environment variable `OLLAMA_HOST` changes that, for example to 0.0.0.0. Wiz Research wrote in 2024 that Ollama's image for Docker is reachable from outside, and that "Ollama does not support authentication out-of-the-box": it asks for no key or token. Wiz found "over 1,000 exposed instances". It also found a hole in Ollama before version 0.1.34 (CVE-2024-37032) through which an attacker could overwrite files on the server, which "can be further exploited into a full Remote Code Execution". Listening on localhost, Wiz wrote, "reduces remote exploitation risk significantly". Cisco's advice of 2025: LLM servers "should never be publicly exposed without requiring secure API key-based or token-based authentication."

    ### Ask for a token

    A server that others must reach can ask for a token, as the APIs in [Batches, keys and the bill](/course/open/build-batch-keys-costs?lang=en) asked for your key. The usual form is the header `Authorization: Bearer` and the token, from RFC 6750 (2012). It calls such a token a bearer token: "any party in possession of the token (a 'bearer') can use the token in any way that any other party in possession of it can." Whoever has it gets in, so it must stay as secret as a key, and the RFC makes TLS, the encryption behind https, "mandatory to implement and use". On the server, the check compares the header with the token it expects.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **attack surface** | the points of a system where an attacker can try to get in, cause an effect or get data out |
        | **secret scanning** | searching code and its history for texts that have the form of a key, a password or a token |
        | push protection | a check before a push that stops a commit with a secret in it |
        | `127.0.0.1`, `localhost` | the address and the name of the computer itself; a server there is reachable only from that computer |
        | `0.0.0.0` | for a server: every address of the computer, also the one the network reaches it on |
        | bearer token | a token that lets in whoever sends it, in the header `Authorization: Bearer` |
        | `rest.isalnum()` | True when the text `rest` is not empty and holds only letters and digits |
        """),
        L.watch_out("Removing a key in a new commit does not remove it from the history, and a key that was ever pushed may already be copied. Revoke it first, then make a new one."),
        mo.Html(C.surface_picture(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_repository():
    mo.md(r"""
    ### The shop's repository

    The course made up a repository for the help assistant of Spoke & Chain, with eight commits. The assistant uses a made-up payment service, PedalPay, whose keys start with `pp_live_`, followed by 24 letters and digits. The course's scanner cuts every file into words, at spaces, new lines and the signs `= " ' ( ) , :`, and asks of each word whether it has the form of a key. The picture shows what it finds in each commit.
    """)
    return


@app.cell(hide_code=True)
def picture_history():
    mo.Html(C.history_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_after_history():
    mo.md(r"""
    In b7d2f41 the key was written into `refunds.py`; one commit later, c90e5a2 read it from the environment instead. The latest version holds no key, but the history still holds two: the second came along in notes on a refund that failed, with the header as it was sent, and was taken out again one commit later. The README has `pp_live_your_key_here`, a text that starts like a key but is not one: the part after `pp_live_` is too short, and has `_` in it. And commit 0a9e4c1 opened a door of another kind: it let the server listen on 0.0.0.0, so that the shop's laptops could use the assistant.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Map the attack surface: the code and its history, the keys, each server and the address it listens on, and the text a model reads.
    2. Keep keys out of the code: read them from an environment variable or a secrets manager.
    3. Scan before every push, and scan the whole history once: look for the forms of the keys you use.
    4. Revoke a key that was ever pushed, then make a new one; taking it out of the code is not enough.
    5. Let a server listen on 127.0.0.1 unless other computers must reach it; if they must, ask for a token, over https.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `C.start_server(token)` starts the course's server: Python's `ThreadingHTTPServer` on `("127.0.0.1", port)`, with a free port, that asks for `token` in the header Authorization. `C.status(url, header)` sends it a request for its models, with `header` in Authorization or with no header for `None`, and gives back the status code of the answer: 200 when it answers, 401 when it refuses. The cell starts the server with the token `C.TOKEN`, sends three requests, and `_server.stop()` stops it again.
    """)
    return


@app.cell
def example_server():
    # the course's server, on this computer only, asking for a token
    _server = C.start_server(C.TOKEN)
    print("listens on", _server.host)
    for _header in [None, "Bearer wrong-token", "Bearer " + C.TOKEN]:
        print(_header, "->", C.status(_server.url, _header))
    _server.stop()
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The server listens on 127.0.0.1, so no other computer can reach it. Without a token, and with a wrong one, it answers 401; only the right token, `shop-token-4417-chain`, gets an answer. Everything ran on your own computer, and the server is stopped again.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_key():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. looks_like_key: the form of a key

        Write `looks_like_key(word)`: True when `word` has the form of a PedalPay key, `pp_live_` and then exactly 24 letters and digits; otherwise False. `word.startswith(start)` gives True when `word` begins with `start`, and `rest.isalnum()` gives True when `rest` is not empty and holds only letters and digits. Replace the two `...`: first give back False when `word` does not start with `pp_live_`; then check `rest`, the part after it. The check calls `looks_like_key(KEY)`, where `KEY` is the key from commit b7d2f41, and then tricky words; the picture under it also runs your function on the shop's history.
        """),
        L.task("Give back True only for pp_live_ followed by exactly 24 letters and digits."),
    ])
    return


@app.cell(hide_code=True)
def exercise_key():
    box_key = L.code_box("looks-like-key", lines=6)
    box_key
    return (box_key,)


@app.cell(hide_code=True)
def check_key(box_key):
    result_key = L.check("looks-like-key", box_key.value)
    result_key
    return (result_key,)


@app.cell(hide_code=True)
def hints_key():
    L.hints(
        "looks-like-key",
        [
            "Step 3 of the recipe: a scanner looks for the form of a key. That form has three parts: the start, the length of the rest, and the characters in it.",
            "The first `...` becomes an `if` with `not` and `word.startswith`, with `return False` under it. The last line gives back whether `rest` has the right number of characters and `rest.isalnum()` holds, with `and` between them.",
            "`    return ___ == 24 and rest.isalnum()`",
        ],
        why="`if not word.startswith(\"pp_live_\"):` with `return False` under it, and at the end `return len(rest) == 24 and rest.isalnum()`. The setup text of the README has 13 characters after `pp_live_`, two of them `_`, so it is not a key.",
    )
    return


@app.cell(hide_code=True)
def text_after_key():
    mo.md(r"""
    Both keys in the history are found, and the setup text of the README is not. A scanner that knows the form of a key needs no list of the keys themselves, but it finds only the forms it knows: a password, or a key of a service whose keys have no form of their own, slips through.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_local():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. local_only: who can reach the server

        Write `local_only(host)`: True when a server that listens on `host` can be reached only from the computer it runs on. `localhost` and `::1`, the name and the address of the computer itself in IPv6, are already in the code. Replace the `...` after `return`: which other addresses belong to the computer itself is in *Open doors* in *The idea*. The check calls `local_only("127.0.0.1")`, then the addresses in the picture under it.
        """),
        L.task("Give back True only for the addresses that keep a server on this computer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_local():
    box_local = L.code_box("local-only", lines=5)
    box_local
    return (box_local,)


@app.cell(hide_code=True)
def check_local(box_local):
    result_local = L.check("local-only", box_local.value)
    result_local
    return (result_local,)


@app.cell(hide_code=True)
def hints_local():
    L.hints(
        "local-only",
        [
            "Step 5 of the recipe, and *Open doors* in *The idea*: which addresses belong to the computer itself?",
            "RFC 1122 makes every address that starts with 127 an address of the computer itself, and `host.startswith(start)` checks how `host` begins. `0.0.0.0` and the empty text are every address at once, so they are not local.",
            "`    return host.startswith(___)`",
        ],
        why="`return host.startswith(\"127.\")`. `0.0.0.0` and `\"\"` give False: a server there listens on every address, also the one the network reaches.",
    )
    return


@app.cell(hide_code=True)
def text_after_local():
    mo.md(r"""
    `10.0.0.127` ends in 127 and is an address on a network; `127.0.0.2` is the computer itself as much as `127.0.0.1` is. `0.0.0.0` and the empty text look like nothing, and are the widest choice of all: every address at once.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_authorized():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. authorized: no token, no entry

        The shop's server reads its token with `os.environ.get("SHOP_TOKEN", "")`, which gives the value of the environment variable `SHOP_TOKEN`, or `""` when it is not set. For every request, it calls `authorized(header, key)` with the header Authorization of the request (`""` when there is none) and that token as `key`. Write `authorized`: True only when `header` is `"Bearer "` followed by the server's token, and never when the server has no token. The last line already compares the header. Replace the `...`. The check calls `authorized("Bearer " + TOKEN, TOKEN)`, where `TOKEN` is the token of the worked example, then other headers, and a server whose token is missing.
        """),
        L.task("Refuse every request when the server has no token; otherwise compare the header."),
    ])
    return


@app.cell(hide_code=True)
def exercise_authorized():
    box_authorized = L.code_box("authorized", lines=5)
    box_authorized
    return (box_authorized,)


@app.cell(hide_code=True)
def check_authorized(box_authorized):
    result_authorized = L.check("authorized", box_authorized.value)
    result_authorized
    return (result_authorized,)


@app.cell(hide_code=True)
def hints_authorized():
    L.hints(
        "authorized",
        [
            "What does the last line give back when the server has no token and the header is `\"Bearer \"`, with nothing after it?",
            "Before the last line comes an `if` on `key`, with `return False` under it. What `key` is when `SHOP_TOKEN` is not set, the text of the exercise says.",
            "`    if key == ___:`",
        ],
        why="`if key == \"\":` with `return False` under it. Without it, `authorized(\"Bearer \", \"\")` gives True: `\"Bearer \" + \"\"` is only `\"Bearer \"`.",
    )
    return


@app.cell(hide_code=True)
def text_after_authorized():
    mo.md(r"""
    Without the first lines, a server started without `SHOP_TOKEN` would let in every request whose header is `Bearer ` with nothing after it: one forgotten setting, and the door is open. With them, a missing token means that nobody gets in, and the mistake shows at once.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Before the push

    Type or paste a commit in the box. The picture runs the scanner on it, line by line, as a check before a push would, with your `looks_like_key` once Exercise 1 is right. Try a key with one letter less, or the setup text of the README. What you type stays on your computer.
    """)
    return


@app.cell(hide_code=True)
def demo_box():
    demo_commit = mo.ui.text_area(value=C.DEMO_TEXT, label="Your commit", rows=5, full_width=True)
    demo_commit
    return (demo_commit,)


@app.cell(hide_code=True)
def demo_push(box_key, demo_commit, result_key):
    _fn = C.mine(box_key.value, result_key.state == "correct", "looks_like_key")
    _out = mo.Html(C.push_picture(demo_commit.value, L.lang, _fn))
    _out
    return


@app.cell(hide_code=True)
def text_exercise_reach():
    mo.md(r"""
    ### Exercise 4. Local, but for whom?

    After commit 1c7f2e5, the assistant's model server listens on 127.0.0.1 again, and asks for no token.
    """)
    return


@app.cell(hide_code=True)
def choice_reach_cell():
    choice_reach = L.choice(
        "reach",
        question="Who can still send it requests?",
        options={
            "right": "Every program on the shop's server itself",
            "nobody": "Nobody but the assistant: one program per address",
            "network": "Every computer on the shop's network",
        },
        correct="right",
        why={
            "right": "Addresses that start with 127 \"MUST NOT appear outside a host\", so other computers cannot reach it; but every program and every user on the same computer can. Wiz wrote that listening on localhost \"reduces remote exploitation risk significantly\": the risk from the network, not all of it. A token also keeps out the programs on the same computer that do not have it.",
            "nobody": "An address does not belong to one program: any program on the computer can send a request to 127.0.0.1. See *Open doors* in *The idea*.",
            "network": "127.0.0.1 is the computer itself, and RFC 1122 says such addresses \"MUST NOT appear outside a host\": other computers cannot reach it. See *Open doors* in *The idea*.",
        },
    )
    choice_reach
    return (choice_reach,)


@app.cell(hide_code=True)
def text_exercise_environment():
    mo.md(r"""
    ### Exercise 5. In the environment, and then?

    The shop now keeps the PedalPay key in the environment variable `PEDALPAY_KEY`, as commit c90e5a2 did, and no longer in the code. When the assistant crashes, the server sends a crash report to an outside service that collects errors, and that report lists the environment of the process.
    """)
    return


@app.cell(hide_code=True)
def choice_environment_cell():
    choice_environment = L.choice(
        "environment",
        question="What does that mean for the PedalPay key?",
        options={
            "right": "Whoever can read the report can use the key now",
            "none": "Nothing: a key outside the code stays out of reports",
            "push": "Push protection holds the report back",
        },
        correct="right",
        why={
            "right": "OWASP warns that \"environment variables are generally accessible to all processes and may be included in logs or system dumps.\" A crash report that lists the environment is such a place: the key goes wherever the report goes. Revoke it, as with a key that was pushed. Out of the code is out of the repository, not out of reach.",
            "none": "OWASP warns that environment variables \"may be included in logs or system dumps\", and a crash report that lists the environment is one of those. See *Keys out of the code* in *The idea*.",
            "push": "Push protection checks what is pushed to a repository on GitHub, not what a server sends elsewhere. See *Secret scanning* in *The idea*.",
        },
    )
    choice_environment
    return (choice_environment,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Injection and secrets](/course/open/rai-classic-vulnerabilities?lang=en) called a key in code a weakness, and [Batches, keys and the bill](/course/open/build-batch-keys-costs?lang=en) read a key from the environment and sent it with `Bearer`. This lesson looked at the other side: how a scanner finds a key that got into the code anyway, and how a server decides whom to let in. The attack surface of an LLM app is more than its prompt. The next lesson, *A model file that runs code*, looks at one more point: the model file you load.

    **An everyday example.** A hotel gives each guest a card that opens one room. When the system that checks the cards breaks down, the doors should stay shut, not open for everyone.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_environment, choice_reach, result_authorized, result_key, result_local):
    L.wrap_up(result_key, result_local, result_authorized, choice_reach, choice_environment)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
