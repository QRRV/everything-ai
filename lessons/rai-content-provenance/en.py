# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-content-provenance. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-content-provenance", "en")
    C = L.checks   # the toy key of 1978, the course's drawings and their labels


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Thinking like an attacker](/course/open/rai-security-basics?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_integrity():
    quick_integrity = L.choice(
        "quick-integrity",
        question="On a news site, someone who has no right to do so changes the caption under a photo. Which property of the CIA triad does that break?",
        options={
            "secret": "Confidentiality: nobody reads what they may not",
            "right": "Integrity: nobody changes what they may not",
            "there": "Availability: it is there when needed",
        },
        correct="right",
        why={
            "right": "Integrity is lost when information is changed or destroyed by someone who has no right to do so. This lesson is about how anyone can tell whether a picture has changed since someone signed it.",
            "secret": "Everyone may read the caption; what went wrong is that someone changed it. See *Three things to protect* in [Thinking like an attacker](/course/open/rai-security-basics?lang=en).",
            "there": "The page is still there for everyone; what went wrong is that someone changed it. See *Three things to protect* in [Thinking like an attacker](/course/open/rai-security-basics?lang=en).",
        },
    )
    quick_integrity
    return (quick_integrity,)


@app.cell(hide_code=True)
def quick_check_spoofing():
    quick_spoofing = L.choice(
        "quick-spoofing",
        question="A picture goes around with a caption saying that a news agency took it, but the agency never did. Which STRIDE threat is that?",
        options={
            "deny": "Repudiation: denying that you did something",
            "right": "Spoofing: pretending to be someone else",
            "rights": "Elevation of privilege: getting rights you should not have",
        },
        correct="right",
        why={
            "right": "Someone puts another's name on the picture. A digital signature works against that: only the holder of the private key can make a signature that checks, as this lesson shows.",
            "deny": "Nobody denies having done something here: someone claims to be the agency. See *What can go wrong: STRIDE* in [Thinking like an attacker](/course/open/rai-security-basics?lang=en).",
            "rights": "Nobody gets more rights than they should: someone claims to be the agency. See *What can go wrong: STRIDE* in [Thinking like an attacker](/course/open/rai-security-basics?lang=en).",
        },
    )
    quick_spoofing
    return (quick_spoofing,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A picture of a flooded harbor goes around, with the name of a news desk under it. Did that desk take it, and is it still the picture the desk took? Looking for signs of a fake is hard. A watermark, a hidden signal in the picture that a detector looks for, can be removed: NIST, the US standards institute, wrote in 2024 that "most current watermarking schemes have been consistently found vulnerable to removal", and that "detecting a watermark always comes with some probability of error".

    Provenance turns the question around. Instead of asking whether a picture is fake, you ask who vouches for it, and whether it has changed since. The maker signs the picture, and anyone can check the signature. This lesson builds that with a key from 1978, small enough to follow by hand, and then looks at C2PA, the standard behind Content Credentials, at what such a label does not tell you, at watermarks, and at what the EU's AI Act asks.

    **An everyday example.** A wax seal on a letter shows whose seal it is, and that nobody opened the letter since. It does not show that what the letter says is true.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Google put a watermark into the answers of its model Gemini, and compared how often about 20 million answers with and without it got a thumbs up. How far apart were the two thumbs-up rates?",
        options={
            "right": "By 0.01%",
            "one": "By about 1%",
            "five": "By about 5%, in favor of no watermark",
        },
        reveal="**By 0.01%.** Dathathri et al., in *Nature* (2024): \"We found that the thumbs-up rate for the two models differed by 0.01% (with the watermarked model being higher); and the thumbs-down rate differed by 0.02% (with the watermarked model being lower).\" Both differences were \"statistically insignificant\". A watermark can leave the answers as good as they were; whether it survives an attacker is another question, which *Watermarks* in *The idea* takes up.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Signing with two keys

    A **digital signature** is, in the words of NIST's standard FIPS 186-5, "The result of a cryptographic transformation of data that, when properly implemented, provides a mechanism for verifying origin authentication, data integrity, and signatory non-repudiation.": it shows who signed, that nothing changed since, and that the signer cannot deny it later. In 1978, Rivest, Shamir and Adleman described how: "A message can be 'signed' using a privately held decryption key. Anyone can verify this signature using the corresponding publicly revealed encryption key." And "An electronic signature must be message-dependent, as well as signer-dependent."

    In their method, RSA, the **public key** is two numbers, `n` and `e`, which anyone may know; the **private key** is one number, `d`, which only the signer has. To sign a number `m`, the signer computes `pow(m, d, n)`. To check a signature `s`, anyone computes `pow(s, e, n)`: if that gives `m` back, the signature fits `m`. In Python, `pow` with three numbers raises the first to the power of the second and gives the remainder after dividing by the third: `pow(2, 10, 1000)` is `24`, since 2 to the power 10 is 1024. The paper's own example is small: "Consider the case p = 47, q = 59, n = p · q = 47 · 59 = 2773, and d = 157." "Therefore e = 17, the multiplicative inverse (mod 2668) of d = 157." Here 2668 is 46 · 58, each prime minus one, multiplied, and being the inverse means that 17 · 157 leaves 1 after dividing by 2668. Signing 920 with this key gives 192, and checking 192 gives 920 back; signing 921 gives 2129; change the signature 192 to 193, and checking gives 1260, not 920. Whoever has only `n` and `e` can check a signature, but cannot sign a number of their choosing: "The security of the system rests in part on the difficulty of factoring the published divisor, n." Whoever finds `p` and `q` from `n` can work out `d`.

    Real keys are far larger. FIPS 186-5, of 2023, asks for "a modulus whose bit length is an even integer and greater than or equal to 2048 bits." The paper's `n` = 2773 has 12 bits, and anyone can find its `p` and `q` in a moment: this lesson's key shows the arithmetic, not security.

    ### Signing a picture

    A picture is not one small number but many bytes. A hash function turns any number of bytes into a value of one fixed length, a fingerprint: change one byte, and the fingerprint changes. The signer signs the fingerprint instead of the picture. C2PA's explainer: "The C2PA specification combines the use of standard cryptographic hashes, such as SHA2-256, with a Merkle tree-like approach". A Merkle tree is a tree of hashes: each hash covers the hashes below it, so the one hash at the top covers every part. In this lesson, `C.toy_hash(picture, n)` takes the SHA-256 of the picture's bytes and keeps its remainder after dividing by `n`, so that it fits the toy key. With only 2773 possible values, two pictures can easily share a toy hash; a real signer signs the whole SHA-256, with a key of 2048 bits or more.

    ### Content Credentials

    **Content provenance** is information about where a piece of content comes from and how it was made or edited. In February 2021, "Founding members Adobe, Arm, BBC, Intel, Microsoft and Truepic" set up the Coalition for Content Provenance and Authenticity, **C2PA**, to "establish a standardized provenance solution with the goal of combating misleading content." Its standard attaches a signed label, a manifest, to a file, such as a picture or a video. **Content Credentials** is the name users see; in the words of the specification: "This is the preferred non-technical term for a C2PA Manifest." The Content Authenticity Initiative, led by Adobe: "They are essentially a nutrition label for digital content".

    A manifest holds assertions, "A data structure which represents a statement either made (or 'created') by the signer or simply gathered at claim generation-time, concerning the asset.", such as which camera or tool made the picture. It holds a claim: "A digitally signed and tamper-evident data structure that references a set of assertions, concerning an asset, and the information necessary to represent the content binding." And it holds the claim signature, made with the signer's private key. The content binding ties the label to the picture. A hard binding is "One or more cryptographic hashes that uniquely identifies either the entire asset or a portion thereof."; a soft binding is a fingerprint that is not unique, or an invisible watermark, which can find a picture again after its label is lost. The Content Authenticity Initiative: "Unlike other types of metadata, Content Credentials are cryptographically signed, making them tamper-evident." Metadata is data about a file, such as when it was made. The same site calls the Leica M11-P "the world's first camera with Content Credentials built in", and contentcredentials.org shows "500+ Companies". The specification this lesson read is version 2.5, of October 2026.

    ### What a label does not tell you

    A label that checks says who signed and that nothing changed since. It does not say that the picture shows what happened. C2PA's explainer: "Content Credentials do not provide value judgments about whether a given set of provenance data is 'true', but instead merely whether the provenance information is well-formed and free from tampering, valid and trusted". "It is not a cure-all for misinformation". NIST adds that C2PA metadata "may be missing, too unspecific to be helpful, or even inaccurate (despite being signed)." Whoever signs a false statement signs it all the same.

    *Valid and trusted* are two things. A signature checks with the key of whoever made it, so a checker has to know whose keys it trusts: in this lesson, `C.KEYS` is that trust list, a dict from a signer's name to its public key. A stranger's signature tells you only that the stranger signed.

    A missing label says little too. C2PA's explainer: "Adding provenance to an asset is optional", and "Can the provenance metadata be removed? Yes it can." NIST: metadata "can similarly be stripped altogether, as it often is when files are shared (e.g., via social media platforms)." That may change one day: "If provenance data becomes widespread, the absence of provenance data in a situation where it would have been expected could also become a signal, at least in some contexts." The aim is to help the honest, not to catch every faker. contentcredentials.org: "While there will continue to be bad actors who seek to label synthetic content as authentic, our goal is to provide good actors with a way to demonstrate the authenticity of their content."

    ### Watermarks

    A **watermark** is a hidden signal in the content itself, which a detector can look for later. Google DeepMind's SynthID "embeds digital watermarks directly into AI-generated images, audio, text or video." A large language model (LLM) writes text one token, a word or part of a word, at a time, and picks each next token by chance from probabilities it computes; that picking is called sampling. For text, "SynthID adjusts these probability scores to generate a watermark": the model picks its next token a little differently, in a way a detector can recognize. Dathathri et al., in *Nature* (2024): "SynthID-Text does not affect LLM training and modifies only the sampling procedure; watermark detection is computationally efficient, without using the underlying LLM." It "has been used to watermark Gemini and Gemini Advanced", and in their measurement on Gemma 7B-IT, an open model of Google, it made each token 0.57% slower to generate.

    Google writes that its image watermark is "designed to stand up to modifications like cropping, adding filters, changing frame rates, or lossy compression." Lossy compression saves space by throwing away detail. The authors of the paper on text are careful about limits: watermarks "do not offer a complete solution to artificial-intelligence text detection; rather they are complementary to other approaches", they are vulnerable "to stealing, spoofing and scrubbing attacks", and "generative watermarks are weakened by edits to the text, such as through LLM paraphrasing". Open models are hard: "enforcing watermarking on these models deployed in a decentralized manner is difficult." NIST: "By having a separate non-watermarked model paraphrase the original watermarked output, it is often possible to remove a text watermark with only minor degradation of text quality."

    A signature and a watermark do different work. The signature says who vouches for a file and that it has not changed since, but it sits in the metadata, which can be removed. The watermark sits in the content and can survive when the metadata is gone, but it can be removed or faked too, and finding it comes with errors. C2PA can carry both: a watermark is one of its soft bindings.

    ### What the law asks

    The EU's AI Act asks for marking. Its Article 50(2): "Providers of AI systems, including general-purpose AI systems, generating synthetic audio, image, video or text content, shall ensure that the outputs of the AI system are marked in a machine-readable format and detectable as artificially generated or manipulated." And: "Providers shall ensure their technical solutions are effective, interoperable, robust and reliable as far as this is technically feasible". Those who deploy an AI system that makes a deepfake, a picture, sound or video made or changed with AI so that it seems real, have to disclose that, by Article 50(4). The Act applies from 2 August 2026; a provider whose system was on the market before that day has until 2 December 2026 to comply with Article 50(2). The text the course read names no technique: not C2PA, and not watermarks.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.Html(C.label_picture(L.lang)),
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **digital signature** | a number that only the holder of a private key can make for a message, and anyone can check with the public key |
        | **public key**, **private key** | `n` and `e`, which anyone may know; `d`, which only the signer has |
        | `pow(m, d, n)` | the remainder of `m` to the power `d` after dividing by `n` |
        | **content provenance** | information about where content comes from and how it was made or edited |
        | **C2PA** | the standard for signed provenance labels, from the coalition of the same name |
        | **Content Credentials** | the name users see for a C2PA manifest: its assertions, its claim and the claim signature |
        | **watermark** | a hidden signal in the content itself, which a detector can look for |
        | `C.toy_hash(picture, n)` | the SHA-256 of the picture's bytes, cut to a number below `n` |
        | `C.BOAT`, `C.KEYS` | the course's drawing of a boat, 108 bytes; the trust list, from a signer's name to its public key |
        """),
        L.watch_out("The key in this lesson has 12 bits, and anyone can break it. Never protect anything real with it, or with signing code of your own: use a library that implements a standard, with keys of 2048 bits or more."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Sign with the private key and check with the public key: only the signer can make a signature, and anyone can check it.
    2. Sign the hash of the content, so that any change to the content breaks the signature.
    3. Check a label in order: is there one, is the signer one you trust, is the content still the one that was signed, and does the signature check?
    4. Read a label that checks as "this signer vouches for it, and it has not changed since", never as "this is true".
    5. Read a missing label as missing, not as fake.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell signs 920, the number in the paper's own example, with the paper's key, and checks the signature.
    """)
    return


@app.cell
def example_sign():
    n, e, d = C.N, C.E, C.D   # the toy key of Rivest, Shamir and Adleman, 1978: 2773, 17 and 157
    m = 920
    s = pow(m, d, n)          # sign with the private exponent d
    print("signature:", s)
    print("checking gives:", pow(s, e, n))
    print("checking 921 with it:", pow(s, e, n) == 921)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Checking gives 920 back, so the signature fits 920; for 921 it does not fit. Signing needed `d`, checking only `e` and `n`. In Exercise 1, numbers arrive with their signatures, and some of the pairs were changed on the way.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_genuine():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. genuine: which signatures check?

        `C.SIGNED` holds five numbers that `photo-desk.example`, a made-up news desk, is said to have signed, each with its signature, as pairs `(m, s)`. Write `genuine(pairs, e, n)`: the numbers `m` from the list `pairs` whose signature `s` checks with the public key `e` and `n`, in the order of `pairs`. `for m, s in pairs:` unpacks each pair into its number and its signature, and the list `found` collects the numbers that check. Replace the `...`. The check calls `genuine(SIGNED, 17, 2773)`, and then a few lists and keys of its own.
        """),
        L.task("Check each signature with the public key, and keep its number when checking gives the number back."),
    ])
    return


@app.cell(hide_code=True)
def exercise_genuine():
    box_genuine = L.code_box("genuine", lines=6)
    box_genuine
    return (box_genuine,)


@app.cell(hide_code=True)
def check_genuine(box_genuine):
    result_genuine = L.check("genuine", box_genuine.value)
    result_genuine
    return (result_genuine,)


@app.cell(hide_code=True)
def hints_genuine():
    L.hints(
        "genuine",
        [
            "*Signing with two keys* in *The idea* says what checking a good signature gives back.",
            "Inside the loop, one `if` decides whether to keep `m`; under it, `found.append(m)`.",
            "`        if ___ == m:`",
        ],
        why="`if pow(s, e, n) == m:` with `found.append(m)` under it. `genuine(SIGNED, 17, 2773)` gives `[920, 1110]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_genuine():
    mo.md(r"""
    Two of the five check. Change the number and keep the signature, and the signature no longer fits; change the signature and keep the number, and it no longer fits either. To sign a number of your own choosing, you need `d`: whoever has only `e` and `n` can check, not sign.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sign():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. sign_picture: a label for a picture

        Write `sign_picture(picture, signer, d, n)`: the label for the bytes `picture`, as a dict with three keys. `"signer"` is the text `signer`, `"hash"` is the picture's hash `toy_hash(picture, n)`, and `"signature"` is the signature of that hash with the private exponent `d`. `toy_hash` is the course's, and the box knows it. Replace the `...`. The check calls `sign_picture(BOAT, "photo-desk.example", 157, 2773)`, and then other pictures, signers and keys.
        """),
        L.task("Hash the picture, sign the hash with d, and give the three parts back as a dict."),
    ])
    return


@app.cell(hide_code=True)
def exercise_sign():
    box_sign = L.code_box("sign-picture", lines=3)
    box_sign
    return (box_sign,)


@app.cell(hide_code=True)
def check_sign(box_sign):
    result_sign = L.check("sign-picture", box_sign.value)
    result_sign
    return (result_sign,)


@app.cell(hide_code=True)
def hints_sign():
    L.hints(
        "sign-picture",
        [
            "First the fingerprint, then the signature: the label signs the hash, not the bytes.",
            "Two lines: one gives `h` its value with `toy_hash`, and one returns a dict with the keys `\"signer\"`, `\"hash\"` and `\"signature\"`.",
            "`    return {\"signer\": signer, \"hash\": h, \"signature\": ___}`",
        ],
        why="`h = toy_hash(picture, n)`, then `return {\"signer\": signer, \"hash\": h, \"signature\": pow(h, d, n)}`. `sign_picture(BOAT, \"photo-desk.example\", 157, 2773)` gives `{\"signer\": \"photo-desk.example\", \"hash\": 1110, \"signature\": 1378}`.",
    )
    return


@app.cell(hide_code=True)
def text_after_sign():
    mo.md(r"""
    That is a toy claim with its claim signature. Change one pixel of the boat and its hash changes, so the old signature no longer fits; to sign the changed picture, you need `d`. A real manifest also signs its assertions, so that nobody can change what it says about the picture without breaking the signature.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_verdict():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. verdict: what does the label say?

        Write `verdict(picture, manifest, keys)`: what the label `manifest` says about the bytes `picture`, as one of five texts. `manifest` is a label such as `sign_picture` makes, or `None` when the picture has none; `manifest is None` asks whether there is none. `keys` is the trust list, a dict from a signer's name to its public key as a pair `(e, n)`, such as `C.KEYS`; `e, n = keys[signer]` unpacks the signer's key, and `signer` is the label's signer. Check in this order: no label gives `"no credentials"`, a signer that is not in `keys` gives `"unknown signer"`, a picture whose own hash differs from the label's `"hash"` gives `"changed"`, a signature that does not check against the label's `"hash"` gives `"bad signature"`, and otherwise the verdict is `"signed by "` followed by the signer. The first lines and the last are there already. Replace the `...`. The check calls `verdict(BOAT, BOAT_LABEL, KEYS)`, and then other pictures, labels and trust lists.
        """),
        L.task("Return a verdict at the first check that fails, in the order of the exercise."),
    ])
    return


@app.cell(hide_code=True)
def exercise_verdict():
    box_verdict = L.code_box("verdict", lines=12)
    box_verdict
    return (box_verdict,)


@app.cell(hide_code=True)
def check_verdict(box_verdict):
    result_verdict = L.check("verdict", box_verdict.value)
    result_verdict
    return (result_verdict,)


@app.cell(hide_code=True)
def hints_verdict():
    L.hints(
        "verdict",
        [
            "Three more checks, each an `if` that returns its verdict: the trust list first, then the picture, then the signature.",
            "`e, n = keys[signer]` comes after the check on the trust list. The two checks after it both compare something with `manifest[\"hash\"]`: first the picture's own hash, then what checking the signature gives.",
            "`    if ___ != manifest[\"hash\"]:`",
        ],
        why="`if signer not in keys:` returns `\"unknown signer\"`; `e, n = keys[signer]`; `if toy_hash(picture, n) != manifest[\"hash\"]:` returns `\"changed\"`; `if pow(manifest[\"signature\"], e, n) != manifest[\"hash\"]:` returns `\"bad signature\"`. `verdict(BOAT, BOAT_LABEL, KEYS)` gives `\"signed by photo-desk.example\"`.",
    )
    return


@app.cell(hide_code=True)
def text_after_verdict():
    mo.md(r"""
    The label of `pixel-mill.example` holds a good signature, made with its own key, but `KEYS` does not hold that key. Add it, and `verdict` says `signed by pixel-mill.example`: whom you trust is your choice, not something the signature decides. The forged label has the right hash for the edited boat, but only the holder of `d` could have signed that hash. And the boat whose label a platform removed is the same boat, pixel for pixel, as the signed one.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: change one pixel

    Pick a pixel of the boat by its number, from 0 at the top left to 107 at the bottom right. The picture flips that pixel: an empty one is filled, a filled one emptied. Beside it stands the course's `verdict` with the boat's label, or with no label when the platform strips it.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_pixel = mo.ui.slider(start=0, stop=107, step=1, value=17, label="Pixel", show_value=True)
    demo_strip = mo.ui.checkbox(value=False, label="The platform strips the label")
    mo.hstack([demo_pixel, demo_strip], justify="start", gap=2)
    return demo_pixel, demo_strip


@app.cell(hide_code=True)
def demo_look(demo_pixel, demo_strip):
    mo.Html(C.demo_picture(demo_pixel.value, demo_strip.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_label():
    mo.md(r"""
    ### Exercise 4. No label

    A picture of a flooded street goes around on a social media platform, without Content Credentials.
    """)
    return


@app.cell(hide_code=True)
def choice_label_cell():
    choice_label = L.choice(
        "no-label",
        question="What does the missing label tell you?",
        options={
            "fake": "That it was made or changed with AI, since a camera would have added one",
            "right": "That it has none now: maybe it never had one, maybe sharing stripped it",
            "real": "That it is real, since tools that make pictures with AI add a label",
        },
        correct="right",
        why={
            "right": "C2PA's explainer: \"Adding provenance to an asset is optional\"; and NIST: metadata is often stripped \"when files are shared (e.g., via social media platforms)\". Only where a label would be expected might its absence one day become a signal, NIST writes.",
            "fake": "Adding a label is optional, and a platform can strip one: in Exercise 3, the boat without its label is pixel for pixel the signed one. See *What a label does not tell you* in *The idea*.",
            "real": "The AI Act asks providers to mark what their systems make, but a mark can be a watermark instead of a label, and a label can be stripped on the way. A missing label points neither way. See *What a label does not tell you* in *The idea*.",
        },
    )
    choice_label
    return (choice_label,)


@app.cell(hide_code=True)
def text_exercise_watermark():
    mo.md(r"""
    ### Exercise 5. Two claims about watermarks

    Google writes that the SynthID watermark in images is "designed to stand up to modifications like cropping, adding filters, changing frame rates, or lossy compression." NIST writes that "most current watermarking schemes have been consistently found vulnerable to removal".
    """)
    return


@app.cell(hide_code=True)
def choice_watermark_cell():
    choice_watermark = L.choice(
        "watermark",
        question="Do the two contradict each other?",
        options={
            "yes": "Yes: NIST's finding covers SynthID too, so Google's claim cannot hold",
            "right": "No: holding up to edits is not holding up to an attacker",
            "text": "No: NIST wrote only about text watermarks, Google about images",
        },
        correct="right",
        why={
            "right": "Holding up to cropping or a filter is not holding up to someone who sets out to remove the watermark. Google claims the first for its image watermark; NIST writes about removal. For text watermarks, the authors of the SynthID-Text paper name \"stealing, spoofing and scrubbing attacks\" themselves, and paraphrasing by an LLM. Ask of a claim of robustness what it was tested against.",
            "yes": "Google does not claim that its watermark survives every attack, only the edits it was designed for; a watermark can do that and still fall to an attacker. Both claims can be true. See *Watermarks* in *The idea*.",
            "text": "NIST's finding is about watermarking schemes in general, not only text. What sets the two claims apart is what they are about: everyday edits, or an attacker. See *Watermarks* in *The idea*.",
        },
    )
    choice_watermark
    return (choice_watermark,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Thinking like an attacker](/course/open/rai-security-basics?lang=en), integrity was something to protect and spoofing a threat; a digital signature lets anyone who has the public key notice either: a changed message, or a signature that is not the named signer's. Signing does not make a picture true: it tells you who vouches for it. The next lesson, *Detecting text written by AI*, asks what is left when nobody signed: telling text written by AI apart by the text alone.

    **An everyday example.** A sealed jar in a shop shows that nobody opened it since the factory, and which factory sealed it. It does not show that what is inside is good; and a jar whose seal fell off has not gone bad for that reason.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_label, choice_watermark, result_genuine, result_sign, result_verdict):
    L.wrap_up(result_genuine, result_sign, result_verdict, choice_label, choice_watermark)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
