# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les math-complex-rotation. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("math-complex-rotation", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_before():
    mo.md(r"""
    **Ook van buiten de course:** de **sinus en cosinus van een hoek**, waarbij het punt bij hoek $\theta$ op de cirkel met straal 1 rond de oorsprong $(\cos\theta, \sin\theta)$ is; en **hoeken in radialen**, waarbij een hele draai $2\pi$ is en een kwartslag $\pi/2$. `np.cos` en `np.sin` verwachten een hoek in radialen, en `np.pi` is $\pi$, net als `math.pi`. Is een van beide weggezakt, lees dan eerst [Eenheidscirkel](https://nl.wikipedia.org/wiki/Eenheidscirkel).
    """)
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een matrix omklappen: de getransponeerde](/course/open/math-transpose?lang=nl) en [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl). <!-- termen:ok -->
    """)
    return


@app.cell(hide_code=True)
def quick_check_matrix():
    quick_matrix = L.choice(
        "quick-matrix",
        question="$\\mathbf{A}$ is de $2 \\times 2$-matrix met rijen $(0, -1)$ en $(1, 0)$. Wat is zijn getransponeerde $\\mathbf{A}^\\top$?",
        options={
            "same": "Dezelfde matrix",
            "right": "Rijen $(0, 1)$ en $(-1, 0)$",
            "minus": "Rijen $(0, 1)$ en $(1, 0)$",
        },
        correct="right",
        why={
            "same": "Alleen een symmetrische matrix is gelijk aan zijn getransponeerde, en hier verschillen de twee getallen naast de diagonaal. Zie [Een matrix omklappen: de getransponeerde](/course/open/math-transpose?lang=nl).",
            "minus": "Transponeren wisselt rijen en kolommen; het haalt geen mintekens weg. Zie [Een matrix omklappen: de getransponeerde](/course/open/math-transpose?lang=nl).",
            "right": "Rij $i$ van $\\mathbf{A}^\\top$ is kolom $i$ van $\\mathbf{A}$. $\\mathbf{A}$ draait een vector een kwartslag tegen de klok in; stap 3 laat zien dat $\\mathbf{A}^\\top$ hem een kwartslag terugdraait.",
        },
    )
    quick_matrix
    return (quick_matrix,)


@app.cell(hide_code=True)
def quick_check_exponent():
    quick_exponent = L.choice(
        "quick-exponent",
        question="Wat is hetzelfde als $e^{a} \\cdot e^{b}$?",
        options={
            "times": "$e^{a \\cdot b}$",
            "right": "$e^{a + b}$",
            "sum": "$e^{a} + e^{b}$",
        },
        correct="right",
        why={
            "times": "Machten met hetzelfde grondtal vermenigvuldigen telt de exponenten op; het vermenigvuldigt ze niet. Zie [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl).",
            "sum": "Een product van twee machten is niet hun som. Zie [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl).",
            "right": "Machten met hetzelfde grondtal vermenigvuldigen telt de exponenten op. Deze les gebruikt die regel voor hoeken.",
        },
    )
    quick_exponent
    return (quick_exponent,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Veel language models geven de plek van elk token aan door paren getallen in zijn query- en keyvector te draaien over een hoek die met de positie groeit. Dat heet rotary position embedding, RoPE; een latere les gaat erover. Een vector met de hand draaien vraagt een rotatiematrix vol sinussen en cosinussen. Complexe getallen doen hetzelfde met één vermenigvuldiging, en ze laten de eigenschap waar het om gaat makkelijk zien: na het draaien speelt van de posities van een query en een key alleen het verschil een rol in hun inproduct. Dezelfde getallen zijn ook de taal van Fourier-analyse, die een signaal in golven opdeelt.
    """)
    return


@app.cell(hide_code=True)
def predict_square():
    guess_square = L.guess(
        "square",
        question="Het complexe getal $1 + i$ is het punt $(1, 1)$: bij 45 graden, op afstand $\\sqrt{2}$ van de oorsprong. Waar komt $(1 + i)^2$ terecht?",
        options={
            "same": "Weer bij 45 graden, op afstand 2",
            "right": "Bij 90 graden, op afstand 2",
            "double": "Bij 90 graden, op afstand $2\\sqrt{2}$",
        },
        reveal="**Bij 90 graden, op afstand 2.** $(1 + i)^2 = 1 + 2i + i^2 = 1 + 2i - 1 = 2i$: het punt $(0, 2)$. Bij het vermenigvuldigen van twee complexe getallen worden hun hoeken opgeteld, $45 + 45 = 90$ graden, en hun lengtes vermenigvuldigd, $\\sqrt{2} \\cdot \\sqrt{2} = 2$.",
    )
    guess_square
    return (guess_square,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Stap 1: het complexe vlak.** Een **complex getal** $z = a + bi$ heeft een reëel deel $a$ en een imaginair deel $b$, met één nieuwe regel: $i^2 = -1$. Teken het als het punt $(a, b)$ in het **complexe vlak**: het reële deel naar rechts, het imaginaire deel omhoog. Zijn **lengte** $|z| = \sqrt{a^2 + b^2}$ is de afstand tot de oorsprong; zijn **hoek** $\varphi$ is zijn richting, tegen de klok in gemeten vanaf de positieve reële as. Met lengte $r$ en hoek $\varphi$ is $z = r(\cos\varphi + i\sin\varphi)$: de **poolvorm**. De **complex geconjugeerde** $\bar z = a - bi$ is het spiegelbeeld van $z$ in de reële as: dezelfde lengte, bij hoek $-\varphi$. En $z\bar z = (a + bi)(a - bi) = a^2 + b^2 = |z|^2$.

        **Stap 2: vermenigvuldigen draait.** Werk uit met $i^2 = -1$: $(a + bi)(c + di) = (ac - bd) + (ad + bc)i$. Neem nu $w = \cos\theta + i\sin\theta$, met lengte 1 en hoek $\theta$:
        """),
        L.formula(r"\begin{aligned} w z &= (a\cos\theta - b\sin\theta) \\ &\quad + (a\sin\theta + b\cos\theta)\,i \end{aligned}"),
        mo.md(r"""
        Het reële en het imaginaire deel zijn precies wat de **rotatiematrix** van de vector $(a, b)$ maakt:
        """),
        L.formula(r"\begin{aligned} &\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} a \\ b \end{pmatrix} \\ &\quad = \begin{pmatrix} a\cos\theta - b\sin\theta \\ a\sin\theta + b\cos\theta \end{pmatrix} \end{aligned}"),
        mo.md(r"""
        Vermenigvuldigen met een complex getal met lengte 1 is dus een **rotatie** over zijn hoek: tegen de klok in bij een positieve hoek. Een getal met lengte $r$ is $r$ keer een getal met lengte 1, dus in het algemeen worden bij het vermenigvuldigen van twee complexe getallen hun lengtes vermenigvuldigd en hun hoeken opgeteld. Uitwerken laat ook zien dat de geconjugeerde van een product het product van de geconjugeerden is: $\overline{zw} = \bar z\,\bar w$.

        **Stap 3: de formule van Euler.** Het getal met lengte 1 bij hoek $\theta$ heeft een korte naam:
        """),
        L.formula(r"e^{i\theta} = \cos\theta + i\sin\theta"),
        mo.md(r"""
        Dit is de **formule van Euler**. Waarom het een macht van $e$ is, vraagt meer analyse dan deze les nodig heeft; hier nemen we hem als gegeven aan en controleren we hem hieronder in NumPy. Hij maakt van de regel uit stap 2 de rekenregel voor exponenten: $e^{i\alpha}\, e^{i\beta} = e^{i(\alpha + \beta)}$, dus draaien over $\alpha$ en daarna over $\beta$ is draaien over $\alpha + \beta$. De geconjugeerde van $e^{i\theta}$ is $\cos\theta - i\sin\theta = e^{-i\theta}$: die draait $\theta$ terug, net als de getransponeerde van de rotatiematrix, zoals $\mathbf{A}^\top$ in de instapcheck. De $n$ getallen $e^{2\pi i k/n}$ met $k = 0, \ldots, n-1$ liggen op gelijke afstanden rond de cirkel, en elk geeft 1 als je het tot de macht $n$ verheft: de **eenheidswortels**. Voor $n = 4$ zijn dat $1$, $i$, $-1$ en $-i$, de kwartslagen. Ze zijn de bouwstenen van de discrete Fourier-transformatie.

        Het plaatje: $z = 2 + i$, en $e^{i\pi/3} z$, met dezelfde lengte en over 60 graden gedraaid. De gestippelde cirkel heeft straal 1.
        """),
        mo.Html(C.plane_picture({"title": "Vermenigvuldigen met e^(iθ) draait", "re": "reëel deel", "im": "imaginair deel", "z": "z = 2 + i", "w": "e^(iπ/3) z", "angle": "60°", "lang": L.lang,
                                 "alt": "Het complexe vlak met een gestippelde cirkel met straal 1. Een pijl van de oorsprong naar z = 2 + i, en een tweede pijl met dezelfde lengte, 60 graden tegen de klok in gedraaid, naar ongeveer 0,13 + 2,23i."})),
        mo.md(r"""
        **Stap 4: een vector als paren.** Een reële vector van even lengte kun je lezen als paren getallen die naast elkaar staan, $(x_0, x_1), (x_2, x_3), \ldots$, en elk paar als één complex getal: $x_0 + i\,x_1$, $x_2 + i\,x_3$, enzovoort. Draai elk paar over zijn eigen hoek, en je hebt de vector stuk voor stuk gedraaid, zonder de lengte van een paar te veranderen. Zo draait RoPE elke query en key.

        **Stap 5: het inproduct na een draaiing.** Bij het inproduct van twee complexe vectoren $\mathbf{u}$ en $\mathbf{v}$ neem je van een van beide de geconjugeerde; deze course neemt die van de tweede: $\sum_j u_j \bar v_j$. Door die geconjugeerde is het inproduct van een vector met zichzelf zijn lengte in het kwadraat, $\sum_j u_j \bar u_j = \sum_j |u_j|^2$, zoals $z\bar z = |z|^2$ in stap 1. Met $\mathbf{u}$ en $\mathbf{v}$ als kolommen is het $\mathbf{v}^H \mathbf{u}$: de rij $\mathbf{v}^H$ is de **geconjugeerde getransponeerde** van $\mathbf{v}$, zijn getransponeerde met elk getal geconjugeerd. Draai nu $\mathbf{u}$ over $\alpha$ en $\mathbf{v}$ over $\beta$. Met de regels uit stap 2 en 3 wordt elke term
        """),
        L.formula(r"\begin{aligned} e^{i\alpha} u_j\, \overline{e^{i\beta} v_j} &= e^{i\alpha} e^{-i\beta}\, u_j \bar v_j \\ &= e^{i(\alpha - \beta)}\, u_j \bar v_j \end{aligned}"),
        mo.md(r"""
        Voor het inproduct telt van de twee draaiingen alleen het verschil $\alpha - \beta$: draai je allebei over dezelfde extra hoek, dan verandert er niets. Dat gebruikt RoPE.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **complex getal** | | een getal $a + bi$, met een **reëel deel** $a$ en een **imaginair deel** $b$ | `3 + 4j`; `z.real`, `z.imag` |
        | $i$ | i | de imaginaire eenheid: $i^2 = -1$ | `1j` |
        | **complexe vlak** | | het vlak waarin $a + bi$ het punt $(a, b)$ is | |
        | $\lvert z\rvert$ | absolute waarde van z | de **lengte** van $z$: de afstand tot de oorsprong | `abs(z)` |
        | $\varphi$ | phi | de **hoek** van $z$: zijn richting, tegen de klok in vanaf de positieve reële as, in radialen | `np.angle(z)` |
        | **poolvorm** | | $z$ geschreven met zijn lengte $r$ en zijn hoek $\varphi$: $r e^{i\varphi}$ | |
        | **complex geconjugeerde**, $\bar z$ | z streep | $a - bi$, het spiegelbeeld van $a + bi$ in de reële as | `np.conj(z)` |
        | **rotatie** | | vectoren om de oorsprong draaien over een hoek, zonder hun lengte te veranderen | |
        | **rotatiematrix** | | de matrix die een rotatie uitvoert; in 2D, over $\theta$, met de rijen $(\cos\theta, -\sin\theta)$ en $(\sin\theta, \cos\theta)$ | |
        | **formule van Euler** | | $e^{i\theta} = \cos\theta + i\sin\theta$ | |
        | $e^{i\theta}$ | e-macht i theta | het complexe getal met lengte 1 bij hoek $\theta$ | `np.exp(1j * theta)` |
        | **eenheidswortel** | | een complex getal $z$ met $z^n = 1$; de $n$ eenheidswortels, $e^{2\pi i k/n}$, liggen op gelijke afstanden rond de cirkel | |
        | **geconjugeerde getransponeerde**, $\mathbf{v}^H$ | v H | de getransponeerde met elk getal geconjugeerd; $\mathbf{v}^H \mathbf{u} = \sum_j u_j \bar v_j$ is het inproduct van complexe vectoren | `np.dot(u, np.conj(v))` |
        """),
        L.watch_out("Python schrijft de imaginaire eenheid als `j`, niet als `i`: `3 + 4j`, en `1j` op zichzelf. Een losse `j` is gewoon een naam."),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De vormen die nieuw zijn: complexe getallen in Python en NumPy, met de lengte en de hoek van $3 + 4i$, de formule van Euler vergeleken met cos en sin, en een kwartslag draaien; en een slice met een stap, die elk tweede getal uit een array neemt.
    """)
    return


@app.cell
def example_complex():
    import numpy as np

    _z = 3 + 4j
    print("reëel", _z.real, " imaginair", _z.imag, " lengte", abs(_z), " hoek", np.angle(_z))
    _theta = np.pi / 2  # een kwartslag, in radialen
    _w = np.exp(1j * _theta)
    print("e^(i theta):", _w, "  cos + i sin:", np.cos(_theta) + 1j * np.sin(_theta))
    print("gedraaid:", _w * _z, " lengte", abs(_w * _z))
    _x = np.array([10, 11, 12, 13, 14, 15])
    print("elk tweede getal, vanaf 0:", _x[0::2], " vanaf 1:", _x[1::2])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    $3 + 4i$ heeft lengte 5 en hoek 0,927 radialen, 53 graden. $e^{i\pi/2}$ komt eruit als `(6.123233995736766e-17+1j)`, hetzelfde als cos plus $i$ sin: dat is $i$, op een afrondingsfout van ongeveer $10^{-16}$ na. Een kwartslag gedraaid wordt $3 + 4i$ gelijk aan $-4 + 3i$, weer met een piepkleine afrondingsfout, en nog steeds met lengte 5. In een slice is een derde getal de stap: `_x[0::2]` begint bij index 0 en neemt elk tweede getal, `[10 12 14]`, en `_x[1::2]` begint bij index 1, `[11 13 15]`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rotate():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een vector paar voor paar draaien

        Schrijf `rotate_pairs(x, angles)`. `x` is een reële NumPy-array van even lengte, gelezen zoals in stap 4: getallen die naast elkaar staan, $(x_0, x_1), (x_2, x_3), \ldots$, zijn paren, en paar $j$ is het complexe getal $x_{2j} + i\,x_{2j+1}$. `angles` is een lijst met één hoek per paar, in radialen. Draai paar $j$ over `angles[j]`, tegen de klok in, en geef een reële array terug met dezelfde indeling als `x`. De check roept `rotate_pairs(np.array([1, 0, 0, 1]), [np.pi / 2, np.pi / 2])` aan, wat `[0, 1, -1, 0]` moet geven, en probeert andere vectoren en hoeken.
        """),
        L.task("Schrijf `rotate_pairs(x, angles)`: draai elk paar getallen naast elkaar in `x` over zijn eigen hoek, en geef een reële array terug met dezelfde indeling."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rotate():
    box_rotate = L.code_box("rotate-pairs", lines=10)
    box_rotate
    return (box_rotate,)


@app.cell(hide_code=True)
def check_rotate(box_rotate):
    result_rotate = L.check("rotate-pairs", box_rotate.value)
    result_rotate
    return (result_rotate,)


@app.cell(hide_code=True)
def hints_rotate():
    L.hints(
        "rotate-pairs",
        [
            "Maak in één keer een complexe vector van alle paren, met de slices uit het uitgewerkte voorbeeld: `x[0::2]` bevat de reële delen, `x[1::2]` de imaginaire. Na het draaien gaat elk reëel deel terug naar een even plek, met zijn imaginaire deel direct erachter.",
        ],
        why="Elk paar wordt één complex getal; vermenigvuldigen met $e^{i\\theta}$ draait het over $\\theta$ en houdt zijn lengte gelijk; de reële en imaginaire delen gaan terug naar waar het paar vandaan kwam. `np.array(angles)` maakt van de lijst hoeken een array, want `1j * angles` werkt niet met een lijst. Een rotatiematrix per paar geeft dezelfde getallen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_relative():
    mo.md(r"""
    ### Oefening 2. Twee draaiingen
    """)
    return


@app.cell(hide_code=True)
def choice_relative_cell():
    choice_relative = L.choice(
        "relative",
        question="Je draait een complexe vector $\\mathbf{q}$ over $3\\theta$ en een complexe vector $\\mathbf{k}$ over $5\\theta$, en neemt hun inproduct $\\sum_j q_j \\bar k_j$. Welke andere twee draaiingen geven altijd hetzelfde inproduct?",
        options={
            "swap": "$\\mathbf{q}$ over $5\\theta$ en $\\mathbf{k}$ over $3\\theta$",
            "right": "$\\mathbf{q}$ over $4\\theta$ en $\\mathbf{k}$ over $6\\theta$",
            "double": "$\\mathbf{q}$ over $6\\theta$ en $\\mathbf{k}$ over $10\\theta$",
        },
        correct="right",
        why={
            "swap": "Dan krijgt elke term de factor $e^{5i\\theta} e^{-3i\\theta} = e^{2i\\theta}$ in plaats van $e^{-2i\\theta}$: een draaiing de andere kant op, en die verandert de som meestal wel. De volgorde telt, omdat van de tweede vector de geconjugeerde wordt genomen. Zie stap 5.",
            "double": "Dan krijgt elke term de factor $e^{6i\\theta} e^{-10i\\theta} = e^{-4i\\theta}$: het verschil is verdubbeld. Alleen het verschil van de hoeken telt, niet hun verhouding. Zie stap 5.",
            "right": "Elke term krijgt de factor $e^{4i\\theta} e^{-6i\\theta} = e^{-2i\\theta}$, net als bij $3\\theta$ en $5\\theta$: beide draaiingen werden $\\theta$ groter, en het verschil bleef $-2\\theta$. Voor RoPE betekent dat: schuiven een query en een key allebei één plek op, dan blijft hun inproduct gelijk.",
        },
    )
    choice_relative
    return (choice_relative,)


@app.cell(hide_code=True)
def text_exercise_real():
    mo.md(r"""
    ### Oefening 3. Het inproduct van reële vectoren
    """)
    return


@app.cell(hide_code=True)
def explain_real():
    explain_real_cell = L.explain(
        "real-dot",
        question="RoPE werkt met reële vectoren en hun gewone inproduct. Lees twee reële vectoren $\\mathbf{x}$ en $\\mathbf{y}$ paar voor paar, zoals in oefening 1, en draai paar $j$ van $\\mathbf{x}$ over $\\alpha_j$ en paar $j$ van $\\mathbf{y}$ over $\\beta_j$. Laat zien dat van de hoeken dan alleen de verschillen $\\alpha_j - \\beta_j$ een rol spelen in hun inproduct $\\mathbf{x} \\cdot \\mathbf{y}$. Noem elke regel die je gebruikt. Begin met één paar: wat is het reële deel van $u \\bar v$, voor $u = a + bi$ en $v = c + di$?",
        model_answer="**Eén paar.** $u \\bar v = (a + bi)(c - di) = (ac + bd) + (bc - ad)\\,i$. Het reële deel, $ac + bd$, is het inproduct van de paren $(a, b)$ en $(c, d)$. Dus $\\mathbf{x} \\cdot \\mathbf{y}$ is het reële deel van $\\sum_j u_j \\bar v_j$, met $u_j$ en $v_j$ paar $j$ van $\\mathbf{x}$ en van $\\mathbf{y}$. **De draaiingen.** Term $j$ wordt $e^{i\\alpha_j} u_j\\, \\overline{e^{i\\beta_j} v_j}$. **Geconjugeerde van een product:** dat is $e^{i\\alpha_j}\\, \\overline{e^{i\\beta_j}}\\, u_j \\bar v_j$, en de geconjugeerde van $e^{i\\beta_j}$ is $e^{-i\\beta_j}$. **Rekenregel voor exponenten:** $e^{i\\alpha_j} e^{-i\\beta_j} = e^{i(\\alpha_j - \\beta_j)}$. Van de hoeken telt in elke term dus alleen $\\alpha_j - \\beta_j$, en daarmee ook in de som en haar reële deel, $\\mathbf{x} \\cdot \\mathbf{y}$. Zijn alle $\\alpha_j = \\beta_j$, dan verandert er niets: beide vectoren op dezelfde manier draaien houdt hun inproduct gelijk.",
        points=[
            "het inproduct van twee paren, $ac + bd$, is het reële deel van $u \\bar v$, dus $\\mathbf{x} \\cdot \\mathbf{y}$ is het reële deel van $\\sum_j u_j \\bar v_j$",
            "je schrijft de gedraaide term $e^{i\\alpha_j} u_j \\cdot \\overline{e^{i\\beta_j} v_j}$ als $e^{i\\alpha_j} e^{-i\\beta_j}\\, u_j \\bar v_j$, met de geconjugeerde van een product en de geconjugeerde van $e^{i\\beta_j}$",
            "je gebruikt $e^{i\\alpha_j} e^{-i\\beta_j} = e^{i(\\alpha_j - \\beta_j)}$, dus in elke term, in de som en in haar reële deel tellen alleen de verschillen",
        ],
    )
    explain_real_cell
    return (explain_real_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De rotatiematrix is een matrix zoals in [Een matrix omklappen: de getransponeerde](/course/open/math-transpose?lang=nl), en zijn getransponeerde draait terug, net als de geconjugeerde; de regel $e^{a} e^{b} = e^{a+b}$ komt uit [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl). De les over rotary position embeddings, RoPE, draait de paren van elke query en key zoals jouw `rotate_pairs` een vector draait: paar $j$ van een token op positie $m$ over $m$ keer een vaste hoek van dat paar. Volgens oefening 3 speelt van de posities $m$ en $n$ van een query en een key dan alleen $m - n$ een rol in hun inproduct. <!-- termen:ok -->
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_relative, explain_real_cell, result_rotate):
    L.wrap_up(result_rotate, choice_relative, explain_real_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
