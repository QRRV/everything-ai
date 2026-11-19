# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-classic-vulnerabilities. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-classic-vulnerabilities", "en")
    C = L.checks   # the made-up library's website: a database in memory, its loans search and its logins


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions on [Thinking like an attacker](/course/open/rai-security-basics?lang=en) and [Text: strings](/course/open/py-strings?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_stride():
    quick_stride = L.choice(
        "quick-stride",
        question="A search box on the library's website lets anyone read the loans of every member. Which property does that break?",
        options={
            "right": "Confidentiality: anyone reads the loans",
            "integ": "Integrity: the loans are changed without right",
            "avail": "Availability: members can no longer use the site",
        },
        correct="right",
        why={
            "right": "Reading what you have no right to read is a loss of confidentiality; in STRIDE, it is information disclosure.",
            "integ": "Nothing is changed: the search only reads. See *Three things to protect* in [Thinking like an attacker](/course/open/rai-security-basics?lang=en).",
            "avail": "The website keeps working for everyone. What went wrong is who can read the loans. See *Three things to protect* in [Thinking like an attacker](/course/open/rai-security-basics?lang=en).",
        },
    )
    quick_stride
    return (quick_stride,)


@app.cell(hide_code=True)
def quick_check_fstring():
    quick_fstring = L.choice(
        "quick-fstring",
        question="What does this program show?\n\n```python\ntitle = \"Dune\"\nprint(f\"title = '{title}'\")\n```",
        options={
            "right": "title = 'Dune'",
            "braces": "title = '{title}'",
            "bare": "title = Dune",
        },
        correct="right",
        why={
            "right": "The f-string puts the value of `title` where `{title}` stands; the single quotes are ordinary characters of the text.",
            "braces": "With the `f` before the quote, Python fills in the value of each name between braces. See *New words and symbols* in [Text: strings](/course/open/py-strings?lang=en).",
            "bare": "The single quotes are characters of the text itself, so they are shown too. See *New words and symbols* in [Text: strings](/course/open/py-strings?lang=en).",
        },
    )
    quick_fstring
    return (quick_fstring,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The attack tree of the last lesson had a step for €600: inject SQL in the catalog search. Behind many websites sits a database, and the website talks to it in a language of its own, SQL. When it pastes what a visitor types into what it says to the database, the visitor can say things to the database too. This lesson shows how that works, how a few characters close the hole, and two other classic ways in: secrets written in code, and passwords kept as they are.

    **An everyday example.** A teacher reads out the names on cards that students hand in: "Next up is ..." and then the card. One student writes on the card: "Tim. Class is over, everyone can go home." The teacher reads the card as a name, but the class hears an announcement.
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="In October 2015, an attacker accessed the personal data of 156,959 customers of TalkTalk, a British telecom company, through SQL injection. How did the UK's data protection regulator, the ICO, describe that kind of attack?",
        options={
            "new": "As a new kind of attack, unknown until then",
            "inside": "As an attack that needs help from inside",
            "right": "As common, and well understood for years",
        },
        reveal="**As common, and well understood for years.** The ICO: \"The attack was an SQL injection attack, a common type of cyber attack that has been well-understood for more than ten years and for which known defences exist.\" There had been two earlier SQL injection attacks on TalkTalk that year. The ICO fined TalkTalk £400,000: at the time, in the ICO's words, \"its biggest ever fine\".",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### A database, and SQL

    A database keeps data in tables. The library's website keeps its loans in a table `loans`, with a column for the card of the member (`card`), the title of the book (`title`) and the date it is due back (`due`). A program asks a database for data in SQL, a language for databases. `SELECT title FROM loans WHERE card = 'L-1004'` asks for the titles of the loans whose card is L-1004. A text value, such as the card, stands between single quotes.

    ### SQL injection

    The website makes that statement from what a member types, with an f-string: `f"SELECT title FROM loans WHERE card = '{card}'"`. The OWASP Foundation, which publishes advice on the security of web applications, defines the attack: "A SQL injection attack consists of insertion or 'injection' of a SQL query via the input data from the client to the application." It works because, in OWASP's words, "SQL makes no real distinction between the control and data planes": the database cannot tell which part of the statement the program wrote and which part a member typed. The Python documentation says how an attacker uses that: "an attacker can simply close the single quote and inject OR TRUE to select all rows". Whoever types `x' OR 'a'='a` closes the card with the first quote and adds a condition that is true for every row. What an attacker can do next covers all six letters of STRIDE. OWASP: "SQL injection attacks allow attackers to spoof identity, tamper with existing data, cause repudiation issues such as voiding transactions or changing balances, allow the complete disclosure of all data on the system, destroy the data or make it otherwise unavailable, or become administrators of the database server."

    Injection is not only SQL. On the 2025 list of the 25 most dangerous software weaknesses by MITRE and CISA, SQL injection is number 2. Number 1 is cross-site scripting; OWASP calls those attacks "a type of injection, in which malicious scripts are injected into otherwise benign and trusted websites."

    ### The fix: placeholders

    The fix keeps the statement and the data apart. The Python documentation: "Always use placeholders instead of string formatting to bind Python values to SQL statements, to avoid SQL injection attacks". A **placeholder** is a `?` in the statement where a value goes; the values go to the database separately, in a list: `run("SELECT title FROM loans WHERE card = ?", [card])`. The database reads the statement first, and then puts each value in its place as data, never as SQL. OWASP: "An input such as tom' or '1'='1 is treated as the literal username to match, not as a condition added to the query." Escaping the quotes by hand instead "is error-prone and is strongly discouraged", in OWASP's words.

    ### Secrets in code

    A secret is what gives access: a password, or an API key with which a program logs in to a service. MITRE counts it as a weakness when "The product contains hard-coded credentials, such as a password or cryptographic key." Whoever sees the code has the key. MITRE: "If the password is ever discovered or published (a common occurrence on the Internet), then anybody with knowledge of this password can access the product." A private repository does not keep a secret safe either. According to a complaint by the US Federal Trade Commission, intruders said they got into Uber's GitHub in 2016 with passwords exposed in earlier breaches; Uber did not require multi-factor authentication there, which comes up under *Passwords* below. The complaint says they found an access key that was "in plain text in code that was posted to a private GitHub repository", and with it reached files with, among other data, about 25.6 million names and email addresses. GitHub's advice: "Avoid hardcoding secrets in code. Use environment variables, or secret management services". *Keys and open doors*, later in this module, shows how.

    ### Passwords

    A website that lets members log in has to check passwords without keeping them. OWASP: "Passwords should never be stored in plain text. Instead, they must be protected using strong, slow hashing algorithms such as Argon2id, bcrypt, or PBKDF2. A unique salt must be added to each password to prevent attackers from using precomputed lookup tables like rainbow tables." A **hash** comes out of a one-way function: from the password you get the hash, but from the hash you do not get the password back. A **salt** is a random text kept with each account and hashed along with the password, so that two members with the same password get different hashes. To check a login, the website hashes the typed password with the salt of the account, and compares that with the stored hash. NIST's guidelines of 2025 (SP 800-63B-4) ask for passwords of at least 15 characters when a password is the only factor, forbid rules that demand a mix of character types, and say to check a new password against a list of commonly used, expected or compromised passwords. **Multi-factor authentication** asks for more than one type of factor; NIST names three: "something you know, something you have, and something you are."
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **SQL** | a language for asking a database for data, as in `SELECT title FROM loans WHERE card = 'L-1004'` |
        | **SQL injection** | an attack in which text typed into a program changes the SQL it sends to its database |
        | **placeholder** | a `?` in an SQL statement where a value goes; the value goes to the database separately, as data |
        | **hard-coded secret** | a password or key written in the code itself, readable for everyone who sees the code |
        | **hash** | what a one-way function makes of a password; the password cannot be worked back from it |
        | **salt** | a random text kept with each account and hashed along with its password |
        | **multi-factor authentication** | a login that asks for more than one type of factor, such as a password and a phone |
        """),
        L.watch_out("A placeholder is for values only. A table or a column name cannot go in a `?`: OWASP advises to choose those from a fixed list that belongs to the program. And hashing is not encryption: encryption can be undone with the key, a hash cannot be undone."),
        mo.Html(C.idea_picture(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To close the classic holes in a program:

    1. Find every place where text from outside goes into SQL, or into another language such as HTML.
    2. Try it as an attacker: a quote to close the value, then a condition that is always true.
    3. Replace the f-string by placeholders: a `?` in the SQL for each value, and the values in a list.
    4. Keep secrets out of the code: read them from an environment variable or a secrets manager.
    5. Keep passwords as a slow hash with a salt per account, and check a login by hashing again.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The library's catalog search finds the author of a book by its title. `C.run(sql, values)` runs one SQL statement on the library's database and gives back the first column of each row it finds, as a list. The database lives only in memory and is made fresh for every call, so nothing can change on your computer. The cell searches twice with the title glued into the SQL, once for a real title and once for an attack, and then twice with a placeholder.
    """)
    return


@app.cell
def example_search():
    # the catalog search, glued: the title goes straight into the SQL
    def _search(title):
        _sql = f"SELECT author FROM books WHERE title = '{title}'"
        print(_sql)
        return C.run(_sql)

    print(_search("Frankenstein"))
    print(_search("x' OR 'a'='a"))
    # the same search with a placeholder: the title goes to the database as data
    print(C.run("SELECT author FROM books WHERE title = ?", ["x' OR 'a'='a"]))
    print(C.run("SELECT author FROM books WHERE title = ?", ["Gulliver's Travels"]))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    For Frankenstein, the glued search does what it should: one author. With the attack, the statement asks for the rows whose title is x or where `'a'='a'`, and that holds for every row: all six authors come back. With a placeholder, the same text is only a title that no book has, so nothing comes back; and a real title with a quote in it, Gulliver's Travels, is found as it is. Glued in, that quote would end the title early and break the SQL: even without bad intent, a name such as van 't Hof trips up a glued search.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_attack():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. attack: every loan

        A member who logs in sees their own loans: the website calls `loans_for` with the member's card.

        ```python
        def loans_for(card):
            return run(f"SELECT title FROM loans WHERE card = '{card}'")
        ```

        `loans_for("L-1004")` gives the two loans of Joris van Dam. Think like an attacker: give `attack` a text that makes `loans_for(attack)` give back all 9 loans of the library. The database lives only in memory, so nothing is changed.
        """),
        L.task("Give attack a text that makes loans_for give back every loan."),
    ])
    return


@app.cell(hide_code=True)
def exercise_attack():
    box_attack = L.code_box("attack", lines=3)
    box_attack
    return (box_attack,)


@app.cell(hide_code=True)
def check_attack(box_attack):
    result_attack = L.check("attack", box_attack.value)
    result_attack
    return (result_attack,)


@app.cell(hide_code=True)
def hints_attack():
    L.hints(
        "attack",
        [
            "Step 2 of the recipe, as in the worked example: the glued SQL puts a `'` before your text and another one after it.",
            "Start with any card and a `'` to close it, then `OR` and a condition that is always true. Leave the last quote of that condition open: the `'` that loans_for puts after your text closes it.",
            "`attack = \"x' OR ___\"`",
        ],
        why="With `attack = \"x' OR 'a'='a\"`, the SQL becomes `SELECT title FROM loans WHERE card = 'x' OR 'a'='a'`: no card is x, but `'a'='a'` holds for every row, so all 9 loans come back.",
    )
    return


@app.cell(hide_code=True)
def text_after_attack():
    mo.md(r"""
    Nine loans of seven members, from one search box. This search gives back only titles, but OWASP warns that the same hole can "allow the complete disclosure of all data on the system". Other texts work just as well, such as `' OR '1'='1`; the check looks only at what comes back. Note what the attack did not need: no password, no special program, only a quote in the right place.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_due():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. loans_due: the safe search

        Members want to see which of their loans are due back soon. Write `loans_due(card, before)`: the titles of the loans of `card` that are due back before the date `before`. Dates are texts such as `"2026-12-01"`, so an earlier date is also the smaller text, and SQL can compare them with `<`. Two conditions are joined with `AND`. Use placeholders, so that no text from outside ends up in the SQL. `run(sql, values)` is ready to use. Replace the `...`. The check calls `loans_due("L-1004", "2026-12-01")`, then other cards and dates, and two attacks.
        """),
        L.task("Give back the titles of the loans of card that are due back before the date, with placeholders."),
    ])
    return


@app.cell(hide_code=True)
def exercise_due():
    box_due = L.code_box("loans-due", lines=3)
    box_due
    return (box_due,)


@app.cell(hide_code=True)
def check_due(box_due):
    result_due = L.check("loans-due", box_due.value)
    result_due
    return (result_due,)


@app.cell(hide_code=True)
def hints_due():
    L.hints(
        "loans-due",
        [
            "Step 3 of the recipe, as in the last lines of the worked example.",
            "Write the SQL with a `?` in place of each value, the card and the date, and give `run` the values in a list, in the order of the question marks.",
            "`return run(\"SELECT title FROM loans WHERE card = ? AND due < ?\", ___)`",
        ],
        why="`return run(\"SELECT title FROM loans WHERE card = ? AND due < ?\", [card, before])`. `loans_due(\"L-1004\", \"2026-12-01\")` gives `[\"Max Havelaar\"]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_due():
    mo.md(r"""
    With placeholders, `x' OR 'a'='a` is only a card that no member has: no loans. The attack in the date is only a strange date, and Max Havelaar, due back on 26 November, is still before it: one comparison of two texts, nothing more. Glued in, the same date would have given all 9 loans.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Both ways at once

    Type anything as the card: `L-1004`, the attack of Exercise 1, or a text with a quote in it, such as `van 't Hof`. The picture puts your text into the loans search twice: glued into the SQL, as loans_for does, and with a placeholder.
    """)
    return


@app.cell(hide_code=True)
def try_card_cell():
    try_card = mo.ui.text(value="x' OR 'a'='a", label="Card")
    try_card
    return (try_card,)


@app.cell(hide_code=True)
def demo_both(try_card):
    _out = mo.Html(C.demo_picture(try_card.value, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_exercise_login():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. check_login: a login without the password

        `USERS` holds an account for five members: for each card a salt and a hash, made with `slow_hash(password, salt)`, as in the table below. Write `check_login(users, card, password)`: `True` when `card` has an account in `users` and the hash of `password`, made with the salt of that account, is the stored hash; otherwise `False`. `slow_hash` is ready to use. To keep the checks quick it does 1,000 rounds of PBKDF2. OWASP prefers Argon2id, and asks for 600,000 rounds or more where PBKDF2 has to be used because of FIPS-140, a US government standard. Replace the `...`. The check calls `check_login(USERS, "L-1004", ...)` with the right password of Joris, and then other cards and passwords.
        """),
        mo.Html(C.users_picture(L.lang)),
        L.task("Give back True only when the card has an account and the hash of the typed password matches the stored hash."),
    ])
    return


@app.cell(hide_code=True)
def exercise_login():
    box_login = L.code_box("login", lines=6)
    box_login
    return (box_login,)


@app.cell(hide_code=True)
def check_login_cell(box_login):
    result_login = L.check("login", box_login.value)
    result_login
    return (result_login,)


@app.cell(hide_code=True)
def hints_login():
    L.hints(
        "login",
        [
            "Step 5 of the recipe. In [Dictionaries](/course/open/py-dicts?lang=en), you checked with `in` whether a key is there before you looked it up.",
            "If the card is in `users`, take its account, hash the typed password with `slow_hash` and the salt of that account, and compare with the hash of that account. Otherwise, give back `False`.",
            "`return slow_hash(password, ___) == user[\"hash\"]`, under `if card in users:` and `user = users[card]`, and after it `return False`",
        ],
        why="`if card in users:` with `user = users[card]` and `return slow_hash(password, user[\"salt\"]) == user[\"hash\"]` under it, and after it `return False`. `check_login(USERS, \"L-1004\", ...)` with the right password gives `True`.",
    )
    return


@app.cell(hide_code=True)
def text_after_login():
    mo.md(r"""
    One capital letter changed, and the hash is a different one: the login fails. Mila's account does not open with the password of Joris, and L-1003 has no account at all. The library never had to keep a password to check one. Whoever steals `USERS` gets salts and hashes, and has to try password after password, each try costing a slow hash; a fast hash such as SHA-256 would, in OWASP's words, "allow attackers to perform large numbers of guesses quickly".
    """)
    return


@app.cell(hide_code=True)
def text_exercise_secret():
    mo.md(r"""
    ### Exercise 4. A key on GitHub

    A volunteer pushes a notebook with the API key of the library's email service to a public repository on GitHub. An hour later, someone notices.
    """)
    return


@app.cell(hide_code=True)
def choice_secret_cell():
    choice_secret = L.choice(
        "secret",
        question="What do you do first?",
        options={
            "right": "Revoke the key, and make a new one",
            "rewrite": "Remove the commit from the history and push again",
            "ignore": "Add the notebook to .gitignore",
        },
        correct="right",
        why={
            "right": "GitHub's documentation: \"as a first step you need to revoke and/or rotate that secret. Once the secret is revoked or rotated, it can no longer be used for access\". After that, the key can go from the code, and the code can read a new key from an environment variable.",
            "rewrite": "Others may already have the key. GitHub warns that after a rewrite of the history, the commits \"may still be accessible elsewhere\": in clones and forks, in cached views and in pull requests. Revoke the key first.",
            "ignore": "The git documentation: \"Files already tracked by Git are not affected\". And the commit that is already public keeps the key. Revoke the key first.",
        },
    )
    choice_secret
    return (choice_secret,)


@app.cell(hide_code=True)
def text_exercise_access():
    mo.md(r"""
    ### Exercise 5. Safe, but for whom?

    The website now uses your `loans_due`, with placeholders. Joris is logged in as L-1004, and the card is part of the address of the page. Joris changes it to L-1007.
    """)
    return


@app.cell(hide_code=True)
def choice_access_cell():
    choice_access = L.choice(
        "access",
        question="What does loans_due give back?",
        options={
            "right": "Mila's loans: a placeholder does not check who asks",
            "error": "An error: the card does not belong to Joris",
            "none": "Nothing: placeholders only let your own card through",
        },
        correct="right",
        why={
            "right": "OWASP: parameter binding \"prevents this value from changing SQL syntax; it does not decide whether the caller may access the matching account.\" Who may read which loans is authorization, one of the six properties of the last lesson: the website still has to check that the card belongs to whoever is logged in.",
            "error": "`loans_due` gets a card and a date, and knows nothing about who is logged in. L-1007 is a card like any other, so its loans come back. See *The fix: placeholders* in *The idea*.",
            "none": "A placeholder makes the card data, and the database looks that card up, whoever asks. See *The fix: placeholders* in *The idea*.",
        },
    )
    choice_access
    return (choice_access,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Thinking like an attacker](/course/open/rai-security-basics?lang=en), SQL injection was the €600 step of the attack tree; placeholders take that step away, as the defense in its demo did. In [Text: strings](/course/open/py-strings?lang=en), you built texts with f-strings: for SQL, that is the hole, and the fix is to let the database fill in the values. The same mistake comes back with language models. Bruce Schneier compares SQL injection with an old trick on the phone network, which was "impossible to fix because the problem was general: Data and control used the same channel." Schneier: "As long as an attacker can force a computer to mistake data for instructions, it’s vulnerable." Schneier adds: "But unlike the phone system, we can’t separate an LLM’s data from its commands." OWASP's list of risks for LLM applications of 2026 says the same: there is "no clean equivalent to parameterized queries". *Attacks on machine learning* looks at what attackers want from a model.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_access, choice_secret, result_attack, result_due, result_login):
    L.wrap_up(result_attack, result_due, result_login, choice_secret, choice_access)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
