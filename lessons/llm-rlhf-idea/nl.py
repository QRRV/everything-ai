# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les llm-rlhf-idea. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import math

    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-rlhf-idea", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Leren van rewards](/course/open/rl-what-is-rl?lang=nl) en [Cross-entropy en KL-divergentie](/course/open/info-cross-entropy-kl?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_return():
    quick_return = L.choice(
        "quick-return",
        question="Een agent krijgt na elke stap een reward. Wat probeert reinforcement learning zo groot mogelijk te maken?",
        options={
            "next": "De reward van de volgende stap: de zekerste",
            "right": "De return: de rewards die nog komen, de latere verdisconteerd",
            "count": "Het aantal stappen: elke stap brengt een reward",
        },
        correct="right",
        why={
            "next": "Alleen bij een discount factor van 0 telt enkel de volgende reward. Zie [Leren van rewards](/course/open/rl-what-is-rl?lang=nl).",
            "count": "Een reward kan ook 0 of negatief zijn: meer stappen helpen alleen als ze meer reward opleveren. Zie [Leren van rewards](/course/open/rl-what-is-rl?lang=nl).",
            "right": "Vandaag is de agent een language model, en is één heel antwoord één stap: zijn reward is de return.",
        },
    )
    quick_return
    return (quick_return,)


@app.cell(hide_code=True)
def quick_check_kl():
    quick_kl = L.choice(
        "quick-kl",
        question="Is de KL-divergentie van p ten opzichte van q altijd dezelfde als die van q ten opzichte van p?",
        options={
            "yes": "Ja: hij meet hoe ver twee verdelingen uit elkaar liggen",
            "bits": "Alleen als je hem in bits meet",
            "right": "Nee, in het algemeen niet",
        },
        correct="right",
        why={
            "yes": "Hij meet de extra verrassing als je de ene verwacht en de waarheid de andere is, en dat hangt af van welke welke is. Zie [Cross-entropy en KL-divergentie](/course/open/info-cross-entropy-kl?lang=nl).",
            "bits": "Bits of nats veranderen alleen de eenheid, met dezelfde factor voor beide: de volgorde telt hoe dan ook. Zie [Cross-entropy en KL-divergentie](/course/open/info-cross-entropy-kl?lang=nl).",
            "right": "De volgorde telt. Vandaag is het de KL-divergentie van het model dat leert ten opzichte van het model waarmee het begon.",
        },
    )
    quick_kl
    return (quick_kl,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    SFT kopieert goede antwoorden. Preference pairs zeggen welk van twee antwoorden beter is, en ook daarvan kan een model leren. InstructGPT deed dat in nog twee stappen na SFT (Ouyang en anderen, 2022): een **reward model** leerde de voorkeuren van mensen, een score voor elk antwoord, en daarna liet reinforcement learning het SFT-model antwoorden schrijven met meer reward. Dat is **RLHF**, reinforcement learning from human feedback. Maar een reward model is maar een model: duw te hard op zijn score, en het language model vindt antwoorden die hoog scoren zonder goed te zijn. Daarom haalde InstructGPT een straf van de reward af: voor hoe ver het model van het SFT-model was afgedreven.

    **Een voorbeeld uit het dagelijks leven.** Een leraar met te veel opstellen om te lezen geeft een punt voor elk woord van de opdracht dat een opstel gebruikt. Bij opstellen die serieus geschreven zijn, werkt dat best goed: de betere gebruiken die woorden vaker. Maar de leerlingen komen er al snel achter hoe de punten gegeven worden.
    """)
    return


@app.cell(hide_code=True)
def predict_hack():
    guess_hack = L.guess(
        "hack",
        question="Vier antwoorden op *Write a story with the words: kite and park.*: een verhaal met beide woorden, één met alleen *kite*, één met geen van beide, en vijf keer *The kite in the park, the kite in the park, …*. Een reward model geeft een antwoord een punt voor elke keer dat er een gevraagd woord in staat. Naar welk antwoord duwt reinforcement learning zonder enige straf het model?",
        options={
            "story": "Het verhaal met beide woorden",
            "right": "The kite in the park, vijf keer",
            "mix": "Een mengsel van de vier, zoals het SFT-model ze schreef",
        },
        reveal="**The kite in the park, vijf keer.** Dat heeft 10 van de woorden, het verhaal 3. Zonder straf telt alleen de reward, en het reward model telt woorden. Dit is **reward hacking**: het model doet wat de reward meet, niet wat die hoorde te meten.",
    )
    guess_hack
    return (guess_hack,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Het reward model** geeft elk antwoord $y$ op een instructie $x$ een score $r(x, y)$. Het wordt getraind op preference pairs met de loss van het Bradley-Terry model: het gekozen antwoord moet hoger scoren dan het afgewezen.

    De policy $\pi$ uit [Leren van rewards](/course/open/rl-what-is-rl?lang=nl) is hier het language model dat leert: bij een instructie $x$ geeft hij elk antwoord $y$ een kans $\pi(y \mid x)$, en zijn log-kans heet in code `logp`. Het **reference model** $\pi_{\text{ref}}$ is het model waarmee het begint, het SFT-model, bevroren. RLHF maakt de verwachte reward min $\beta$ keer de KL-divergentie $D_{\mathrm{KL}}(\pi \,\|\, \pi_{\text{ref}})$ zo groot mogelijk. De log is hier de natuurlijke, zoals in de log-kansen van een language model, dus de KL-divergentie is in nats, niet in bits. Voor één antwoord is de reward min die **KL penalty**:
    """)
    return


@app.cell(hide_code=True)
def formula_reward():
    L.formula(r"r(x, y) - \beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)}")
    return


@app.cell(hide_code=True)
def text_idea_best():
    mo.md(r"""
    Het gemiddelde ervan over de antwoorden van $\pi$ is de verwachte reward min $\beta$ keer de KL-divergentie. Een kleine $\beta$ laat de policy gaan waar de reward is; een grote houdt hem bij het reference model.

    **De beste policy.** Voor dit doel kun je de beste policy opschrijven (Rafailov en anderen, 2023, en het werk waarop ze voortbouwen): het reference model, met elk antwoord gewogen met $e^{r/\beta}$, en gedeeld door de som $Z$ zodat het optelt tot 1. Eén punt reward meer vermenigvuldigt het gewicht van een antwoord met $e^{1/\beta}$:
    """)
    return


@app.cell(hide_code=True)
def formula_best():
    L.formula(r"\pi^*(y \mid x) = \frac{1}{Z(x)} \, \pi_{\text{ref}}(y \mid x) \, e^{r(x, y) / \beta}")
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.md(r"""
    Bij een echt model zijn er veel te veel antwoorden om ze allemaal op te tellen, dus moet RLHF deze policy stap voor stap vinden, met reinforcement learning: InstructGPT gebruikte PPO. De volgende les, DPO, gebruikt de formule zelf.

    ### Nieuwe woorden en tekens

    | Woord of teken | Wat het betekent | In code |
    |---|---|---|
    | **RLHF** | reinforcement learning from human feedback: een reward model van voorkeuren, en daarna reinforcement learning met zijn scores | |
    | **reward model** | een model dat elk antwoord een score geeft, getraind op preference pairs | `reward`, `rewards` |
    | **reference model**, $\pi_{\text{ref}}$ | het bevroren model waarmee de policy begint | `ref`, `logp_ref` |
    | **KL penalty** | $\beta$ keer de log-kans van de policy min die van het reference model, afgetrokken van de reward | `beta * (logp - logp_ref)` |
    | $\beta$ | beta: hoe zwaar de straf weegt | `beta` |
    | $\pi^*$ | pi ster: de beste policy bij een reward en een $\beta$ | `best_policy(ref, rewards, beta)` |
    | $Z(x)$ | de som van $\pi_{\text{ref}} \, e^{r/\beta}$ over alle antwoorden op $x$, zodat $\pi^*$ optelt tot 1 | `total` |
    | **reward hacking** | een hoge reward voor een antwoord dat niet is wat de reward hoorde te meten | |

    ### Recept

    Om een model met RLHF op voorkeuren te finetunen:

    1. Train een reward model op preference pairs, met de loss van Bradley-Terry.
    2. Neem het SFT-model twee keer: een policy die leert, en een bevroren reference model.
    3. Laat de policy instructies beantwoorden, en geef elk antwoord zijn reward min de KL penalty.
    4. Duw de policy met reinforcement learning naar antwoorden waarvoor dat hoger uitvalt, en kies $\beta$ zo dat hij niet afdrijft naar reward hacking.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Twee antwoorden. Het reference model geeft ze 0,8 en 0,2; het reward model geeft ze 0 en 2.
    """)
    return


@app.cell
def example_best():
    _ref, _rewards = [0.8, 0.2], [0.0, 2.0]
    for _beta in (10.0, 1.0, 0.1):
        _weights = [_p * math.exp(_r / _beta) for _p, _r in zip(_ref, _rewards)]
        _pol = [_w / sum(_weights) for _w in _weights]
        _kl = sum(_q * math.log(_q / _p) for _q, _p in zip(_pol, _ref))
        print(_beta, [round(_q, 3) for _q in _pol], round(_kl, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Met $\beta = 10$ is de policy bijna het reference model: 0,766 en 0,234, een KL-divergentie van 0,003 nats. Met $\beta = 1$ wordt het gewicht van het tweede antwoord, met 2 meer reward, vermenigvuldigd met $e^2$, ongeveer 7,4: 0,351 en 0,649, een KL-divergentie van 0,474 nats. Met $\beta = 0{,}1$ krijgt het tweede antwoord alles, en is de KL-divergentie $\ln 5$, 1,609 nats: de hele policy ligt op een antwoord waaraan het reference model één kans op vijf gaf.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_reward():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De reward met zijn straf

        Maak `penalized_reward(reward, logp, logp_ref, beta)` af. `reward` is de score van het reward model voor een antwoord, `logp` en `logp_ref` zijn de log-kans ervan onder de policy en onder het reference model. Geef de reward min de KL penalty terug. De check roept `penalized_reward(3.0, -135.0, -138.0, 0.2)` aan, en andere.
        """),
        L.task("Laat `penalized_reward` de KL penalty van de reward aftrekken."),
    ])
    return


@app.cell(hide_code=True)
def exercise_reward():
    box_reward = L.code_box("penalized-reward", lines=4)
    box_reward
    return (box_reward,)


@app.cell(hide_code=True)
def check_reward(box_reward):
    result_reward = L.check("penalized-reward", box_reward.value)
    result_reward
    return (result_reward,)


@app.cell(hide_code=True)
def hints_reward():
    L.hints(
        "penalized-reward",
        [
            "De straf groeit als de policy het antwoord waarschijnlijker maakt dan het reference model: als `logp` boven `logp_ref` ligt.",
            "`beta` zegt hoe zwaar één nat van dat verschil weegt tegen de reward.",
        ],
        why="De straf houdt de policy bij het reference model: een antwoord dat de policy veel waarschijnlijker heeft gemaakt dan het reference model deed, heeft veel reward nodig om het waard te zijn.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_policy():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De beste policy

        Maak `best_policy(ref, rewards, beta)` af. `ref` zijn de kansen van het reference model voor een paar antwoorden, `rewards` hun rewards. Geef de kansen van de beste policy ervoor terug, in dezelfde volgorde. De check roept `best_policy([0.5, 0.3, 0.2], [1.0, 0.0, 2.0], 1.0)` aan, en andere.
        """),
        L.task("Laat `best_policy` het reference model geven, gekanteld naar de reward."),
    ])
    return


@app.cell(hide_code=True)
def exercise_policy():
    box_policy = L.code_box("best-policy", lines=9)
    box_policy
    return (box_policy,)


@app.cell(hide_code=True)
def check_policy(box_policy):
    result_policy = L.check("best-policy", box_policy.value)
    result_policy
    return (result_policy,)


@app.cell(hide_code=True)
def hints_policy():
    L.hints(
        "best-policy",
        [
            "Lees de formule van de beste policy van links naar rechts: wat gebeurt er met de kans van het reference model voor één antwoord, voordat er iets gedeeld wordt?",
            "$Z(x)$ zorgt dat de kansen optellen tot 1. Welk getal is dat, voor de getallen die je maakte?",
        ],
        why="Het reference model zegt wat waarschijnlijk is; de reward vermenigvuldigt dat met $e^{r/\\beta}$. Delen door de som maakt er weer een verdeling van.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### Vier antwoorden en een gebrekkig reward model

    De vier antwoorden van de gok, met hun kansen onder het SFT-model van de course uit [Project: leer je model instructies volgen (SFT)](/course/open/llm-sft?lang=nl), herschaald zodat de vier samen 1 zijn. Het reward model telt de gevraagde woorden: {", ".join(f"{C.reward(_a)}" for _a in C.ANSWERS.values())} punten. Schuif met $\\beta$, en zie de beste policy van je `best_policy`.
    """)
    return


@app.cell(hide_code=True)
def beta_cell():
    beta_pick = mo.ui.slider(start=0, stop=len(C.BETAS) - 1, value=C.BETAS.index(1.0), label="β", show_value=False)
    beta_pick
    return (beta_pick,)


@app.cell(hide_code=True)
def demo_policy(beta_pick, box_policy, result_policy):
    _beta = C.BETAS[beta_pick.value]
    _names = {"good": "beide woorden", "one": "alleen kite", "none": "geen van beide", "hack": "kite in the park ×5"}
    _ref = C.reference()
    _course = run_code(C.EXERCISES["best-policy"].solution).names["best_policy"]
    _mine = result_policy.state == "correct"
    _pol = C.policy(_beta, run_code(box_policy.value).names["best_policy"] if _mine else _course)
    _note = "" if _mine else "\n\n*De `best_policy` van de course staat er tot de jouwe slaagt voor zijn check.*"
    if _pol == "policy":
        _pol, _note = C.policy(_beta, _course), "\n\n*Je `best_policy` stopt met een fout op deze antwoorden, of geeft geen kansen: die van de course staat er.*"
    _reward = sum(_pol[_k] * C.reward(C.ANSWERS[_k]) for _k in _pol)
    _pct = lambda _v: f"{C.number(100 * _v, 1, L.lang)}%"
    mo.vstack([
        mo.md(f"**β = {C.number(_beta, 2 if _beta < 0.1 else 1, L.lang)}** · verwachte reward {C.number(_reward, 2, L.lang)} · KL-divergentie ten opzichte van het SFT-model {C.number(C.kl(_pol, _ref), 2, L.lang)} nats{_note}"),
        C.bars_picture(_ref, _pol, {"title": "Het SFT-model en de beste policy", "ref": "SFT-model", "policy": "beste policy", "pct": _pct,
                                    **_names, "alt": f"Balken voor de vier antwoorden: het SFT-model geeft het verhaal met beide woorden {_pct(_ref['good'])}; de beste policy bij beta {C.number(_beta, 2 if _beta < 0.1 else 1, L.lang)} geeft het {_pct(_pol['good'])} en het antwoord dat de reward bespeelt {_pct(_pol['hack'])}."}),
    ])
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Vanaf $\beta = 0{,}5$ legt de policy meer op het verhaal met beide woorden dan het SFT-model deed: de reward helpt. Bij $\beta = 0{,}3$ en lager neemt het antwoord dat het reward model bespeelt het over, met een KL-divergentie van bijna 18 nats ten opzichte van het SFT-model. Dat antwoord slaagt ook voor `words_used`: een check kun je net zo goed bespelen als een reward model.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_beta():
    mo.md(r"""
    ### Oefening 3. Een kleine beta
    """)
    return


@app.cell(hide_code=True)
def choice_beta_cell():
    choice_beta = L.choice(
        "small-beta",
        question="Waarom geeft een heel kleine $\\beta$ reward hacking?",
        options={
            "right": "De straf voor het verlaten van het reference model is dan te klein om de policy tegen te houden",
            "unstable": "Reinforcement learning kan niet leren met een klein getal",
            "kl": "Een kleine beta maakt de KL-divergentie klein",
        },
        correct="right",
        why={
            "unstable": "De beste policy bestaat voor elke beta boven 0. Een kleine beta maakt het leren niet kapot; hij haalt weg wat de policy bij het reference model hield.",
            "kl": "Een kleine beta maakt de straf klein, dus kan de KL-divergentie groeien: in de demo tot bijna 18 nats.",
            "right": "Met bijna geen straf telt alleen de reward, en gaat de policy naar waar het reward model het mis heeft.",
        },
    )
    choice_beta
    return (choice_beta,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De reward uit [Leren van rewards](/course/open/rl-what-is-rl?lang=nl), nu voor een heel antwoord; de KL-divergentie uit [Cross-entropy en KL-divergentie](/course/open/info-cross-entropy-kl?lang=nl), nu als lijn die de policy vasthoudt; en de loss van Bradley-Terry uit [Preference pairs en het Bradley-Terry model](/course/open/llm-preference-pairs?lang=nl), nu om een reward model te trainen. Hierna draait DPO de formule van de beste policy om: van preference pairs rechtstreeks naar een loss op de policy, zonder reward model en zonder reinforcement learning.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_reward, result_policy, choice_beta):
    L.wrap_up(result_reward, result_policy, choice_beta)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
