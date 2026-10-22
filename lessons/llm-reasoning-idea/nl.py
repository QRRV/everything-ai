# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les llm-reasoning-idea. Route: course/route/lessen/taal.yaml. Checks en de run van de course: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-reasoning-idea", "nl")
    C = L.checks
    R = C.course()      # de run van de course: Qwen3-0.6B-Base op de eerste 30 testvragen van GSM8K
    S = C.SCORES        # wat die run zegt, gelezen en gestemd met de eigen oplossingen van de course
    Q1 = R["questions"][0]


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Heeft post-training geholpen?](/course/open/llm-eval-chat?lang=nl) en [Hoe zeker is een getal? Betrouwbaarheidsintervallen en de bootstrap](/course/open/stats-bootstrap?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_program():
    _quick_program = L.choice(
        "quick-program",
        question="Wat besliste in *Heeft post-training geholpen?* of een verhaal deed wat de instructie vroeg?",
        options={
            "right": "Een programma dat het nakijkt, zoals of een woord wegblijft",
            "reward": "Een reward model dat elk verhaal een score geeft",
            "people": "Jij, door elk verhaal te lezen",
        },
        correct="right",
        why={
            "right": "De instructies waren verifieerbaar: een programma keek elk antwoord na, voor elk model op dezelfde manier.",
            "reward": "Er was geen reward model: je eigen functies keken elk antwoord na. Zie [Heeft post-training geholpen?](/course/open/llm-eval-chat?lang=nl).",
            "people": "Je functies keken alle 600 antwoorden na. Zie [Heeft post-training geholpen?](/course/open/llm-eval-chat?lang=nl).",
        },
    )
    _quick_program
    return


@app.cell(hide_code=True)
def quick_check_error():
    _quick_error = L.choice(
        "quick-error",
        question="Een accuracy gemeten op 30 vragen heeft een standaardfout van ongeveer 9 punten. Op ongeveer hoeveel vragen is dat 3 punten?",
        options={
            "three": "90",
            "right": "270",
            "thirty": "900",
        },
        correct="right",
        why={
            "right": "De standaardfout krimpt met de wortel uit het aantal vragen: een derde ervan vraagt negen keer zoveel vragen.",
            "three": "Drie keer zoveel vragen deelt hem door de wortel uit 3, niet door 3. Zie [Hoe zeker is een getal?](/course/open/stats-bootstrap?lang=nl).",
            "thirty": "Dertig keer zoveel vragen deelt hem door de wortel uit 30, ongeveer 5,5. Zie [Hoe zeker is een getal?](/course/open/stats-bootstrap?lang=nl).",
        },
    )
    _quick_error
    return


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Elk token dat een model schrijft, komt uit één forward pass door zijn blokken, 28 in Qwen3-0.6B. Een vraag die een paar stappen werk vraagt, zoals een redactiesom met drie berekeningen, krijgt dan niet meer van dat werk dan een vraag die er één vraagt, als het model meteen antwoordt. Mensen beantwoorden zulke vragen ook niet meteen: ze schrijven de stappen op.

    **Een voorbeeld uit het dagelijks leven.** 47 × 38 uit je hoofd, tegenover op papier. Op papier doe je steeds maar één kleine stap, en je leest terug wat je schreef.
    """)
    return


@app.cell(hide_code=True)
def predict_steps():
    _guess_steps = L.guess(
        "steps",
        question=f"Qwen3-0.6B-Base, een klein open base model, beantwoordt de eerste {S['n']} sommen van GSM8K, de benchmark van OpenAI met redactiesommen van de basisschool (licentie MIT). Met vier voorbeelden die alleen hun antwoord laten zien, heeft het er {S['direct']} goed. Met dezelfde vier voorbeelden en hun uitgeschreven stappen: hoeveel?",
        options={
            "same": "Ongeveer evenveel: het is hetzelfde model",
            "few": "Een paar meer, tot twee keer zoveel",
            "half": "Ongeveer de helft",
        },
        reveal=f"**{S['steps']} van de {S['n']}** in de run van de course: dezelfde weights, dezelfde vragen, alleen de prompt is anders. Het Qwen3 Technical Report (2025) geeft {C.number(C.QWEN3_REPORT, 1, 'nl')}% op alle 1.319 testsommen met zo'n prompt.",
    )
    _guess_steps
    return


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(rf"""
        ## Het idee

        **Eén token, één forward pass.** Het directe antwoord van Qwen3 op vraag 1 is *{Q1['direct']['text'].strip()}* Dat zijn {Q1['direct']['tokens']} tokens, dus {Q1['direct']['tokens']} forward passes, en het is fout. Het antwoord met stappen is er {Q1['steps']['tokens']}, waarbij elke pass via attention alles leest wat er al staat, en eindigt bij het goede antwoord, {Q1['answer']}. Stappen geven meer rekenwerk voor dezelfde vraag (in theorie kan een transformer van vaste diepte sommige taken niet in één pass, en met genoeg uitgeschreven stappen wel), en kleinere problemen, die elk in één pass te doen zijn.
        """),
        L.watch_out("*Redeneert* is beeldspraak. Het model voorspelt nog steeds het ene token na het andere: zijn stappen zijn tekst die het schreef en daarna leest."),
        mo.md(rf"""
        **Drie manieren om de stappen te krijgen.** Uitgewerkte voorbeelden in de prompt: few-shot prompting met een chain of thought. Erom vragen zonder voorbeelden: zero-shot, *Let's think step by step.* na de vraag, het experiment uit [Hoe onderzoek werkt](/course/open/res-research-cycle?lang=nl); in de run van de course {S['zero']} van de {S['n']} goed. Of het model trainen om ze te schrijven: reasoning models zoals OpenAI o1 en DeepSeek-R1 (stand oktober 2026) leerden het met reinforcement learning, DeepSeek-R1 op antwoorden die een programma nakijkt, zoals in [Reinforcement learning met verifiable rewards](/course/open/llm-rlvr?lang=nl).

        **Test-time compute.** Stappen schrijven kost meer rekenwerk tijdens het antwoorden, en meerdere antwoorden ook. In de run van de course was een antwoord gemiddeld {C.number(S['tokens']['direct'], 1, 'nl')} tokens direct en {C.number(S['tokens']['steps'], 0, 'nl')} met stappen, en {C.SAMPLES} samples samen {C.number(C.SAMPLES * S['tokens']['sample'], 0, 'nl')}: op de CPU van de course ongeveer {C.number(S['seconds']['direct'], 0, 'nl')} s, {C.number(S['seconds']['steps'], 0, 'nl')} s en {C.number(S['seconds']['votes'] / 60, 0, 'nl')} minuten per vraag.

        **Stemmen en pass@k.** Sample $k$ antwoorden bij een temperature boven 0, lees van elk het eindgetal, en neem het getal dat het vaakst komt: majority voting, dezelfde stemming als in [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl), bij chains of thought ook self-consistency genoemd. Pass@k vraagt iets anders: is er *minstens één* van de $k$ goed? Daarvoor is een check nodig die het antwoord kent, hier het goede antwoord uit GSM8K; stemmen heeft er geen nodig.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent |
        |---|---|
        | **chain of thought** | de stappen die een model schrijft vóór zijn eindantwoord |
        | **few-shot prompting** | een paar uitgewerkte voorbeelden in de prompt, vóór de echte vraag |
        | **zero-shot prompting** | de vraag of een instructie, zonder voorbeelden |
        | **test-time compute** | het rekenwerk dat een model besteedt terwijl het antwoordt |
        | **pass@k**, $k$ | of minstens één van $k$ antwoorden goed is, volgens een check die het antwoord kent |
        | **reasoning model** | een model dat, meestal met reinforcement learning, getraind is om een lange chain of thought te schrijven voordat het antwoordt |
        | **faithfulness** | of de uitgeschreven stappen de echte reden voor het antwoord zijn |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play_questions():
    mo.md(rf"""
    **De run van de course, vraag voor vraag.** Qwen3-0.6B-Base op de eerste {S['n']} testvragen van GSM8K, geen uitgezochte, met beide prompts, greedy. De balken zijn de tokens van elk antwoord, met ✓ of ✗ van de checker van de course; daaronder staan beide antwoorden voluit.
    """)
    return


@app.cell(hide_code=True)
def play_questions():
    _read = C.course_functions()[0]
    _labels = {"title": "Vraag {i}: de tokens van elk antwoord", "x": "geschreven tokens", "direct": "direct", "steps": "met stappen",
               "tokens": "tokens", "read": "gelezen:", "nothing": "niets", "key": "het goede antwoord is", "question": "Vraag {i} van {n}",
               "alt": "De tokens van het directe antwoord en van het antwoord met stappen op de gekozen vraag, als twee balken, elk met een vinkje als de checker van de course het goede getal vond en een kruisje als dat niet zo is."}
    _play = L.explore("questions", "vraag", lambda _v: C.answer_frame(R, int(_v), _read, _labels), start=1, low=1, high=S["n"], step=1,
              role="", task="Zoek een vraag waarbij het directe antwoord goed is en de chain of thought fout. Waar gaat de chain mis?")
    _play
    return


@app.cell(hide_code=True)
def text_play_questions_after():
    mo.md(rf"""
    Vraag {C.COUNTER + 1} is er een: het directe antwoord, {R['questions'][C.COUNTER]['answer']}, is goed, en de chain of thought schrijft *60 \* 3 = 180 meters each week*: hij vergeet dat James elke keer 3 sprints loopt. Uitgeschreven stappen helpen bij de meeste vragen, niet bij alle, en een chain of thought kan overtuigend klinken en toch fout zijn.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Schrijf een prompt van een paar uitgewerkte voorbeelden, elk `Question:`, `Answer:` met de stappen, `The answer is N.` en een lege regel; eindig met de nieuwe vraag en `Answer:`.
    2. Laat het model schrijven tot het een eigen vraag begint of stopt; greedy voor één antwoord.
    3. Lees zijn eindgetal met een programma: het getal na de eerste `The answer is`, anders het laatste getal.
    4. Om te stemmen: sample $k$ antwoorden bij temperature 0,7, lees ze allemaal, neem het getal dat het vaakst komt.
    5. Vergelijk met het goede antwoord; kijk voor pass@k of er minstens één van de $k$ goed is.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 1 met één voorbeeld, en stap 3 tot en met 5 op de antwoorden van de course op vraag 1. `C.examples()` geeft de vier voorbeelden van de les, elk een dict met `"question"`, `"steps"` en `"answer"`. Voer de cel uit.
    """)
    return


@app.cell
def example_question_1():
    _example = C.examples()[0]
    _q = R["questions"][0]
    # 1. één uitgewerkt voorbeeld met zijn stappen, dan de nieuwe vraag
    _prompt = "Question: " + _example["question"] + "\nAnswer: " + _example["steps"] + "\nThe answer is " + _example["answer"] + ".\n\n"
    _prompt = _prompt + "Question: " + _q["question"] + "\nAnswer:"
    # de prompt regel voor regel: repr zet elke regel tussen aanhalingstekens, dus een lege regel is ''
    for _line in _prompt.split("\n"):
        print(repr(_line))
    # 3. het getal na de eerste "The answer is", in het directe antwoord en in dat met stappen
    for _kind in ["direct", "steps"]:
        _after = _q[_kind]["text"].split("The answer is")[1]
        print(_kind, repr(_after), "->", _after.split()[0].strip("."))
    # 4. wat de checker in de 8 samples las, geteld
    _read = ["0", "26", "18", "18", "26", "18", "18", "224"]
    _counts = {}
    for _a in _read:
        if _a in _counts:
            _counts[_a] = _counts[_a] + 1
        else:
            _counts[_a] = 1
    print(_counts)
    # 5. het goede antwoord
    print("goede antwoord:", _q["answer"])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(rf"""
    `repr` zet elke regel van de prompt tussen aanhalingstekens, dus de lege regel na het voorbeeld is `''`: daar staat in de prompt `\n\n`, twee nieuwe regels achter elkaar. De laatste regel is `'Answer:'`, waar het model verdergaat. In het directe antwoord leest de checker {C.course_functions()[0](Q1['direct']['text'])} en in dat met stappen {C.course_functions()[0](Q1['steps']['text'])}; het goede antwoord is {Q1['answer']}. Van de {C.SAMPLES} samples komt {C.FIRST_VOTE['top']} het vaakst, {C.FIRST_VOTE['count']} keer: de stemming geeft {C.FIRST_VOTE['top']}, en dat is goed; en omdat minstens één sample goed is, is pass@{C.SAMPLES} dat ook.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_prompt():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. Een prompt met uitgewerkte voorbeelden

        Maak `few_shot_prompt(examples, question, steps)` af: `examples` is een lijst van dicts zoals `C.examples()` ze geeft. Is `steps` waar, dan staat elk voorbeeld zoals in het uitgewerkte voorbeeld, met een lege regel erna; is `steps` niet waar, dan zonder de stappen en hun nieuwe regel: `Answer: The answer is 72.` Daarna `Question: `, de nieuwe vraag, een nieuwe regel en `Answer:`, zonder spatie erna. De check probeert de voorbeelden van de les met en zonder stappen, een eigen voorbeeld, en stappen van meer regels.
        """),
        L.task("Maak `few_shot_prompt` af."),
    ])
    return


@app.cell(hide_code=True)
def exercise_prompt():
    box_prompt = L.code_box("few-shot-prompt", lines=12)
    box_prompt
    return (box_prompt,)


@app.cell(hide_code=True)
def check_prompt(box_prompt):
    result_prompt = L.check("few-shot-prompt", box_prompt.value)
    result_prompt
    return (result_prompt,)


@app.cell(hide_code=True)
def hints_prompt():
    L.hints(
        "few-shot-prompt",
        ["Bouw een lijst van teksten, één voorbeeld tegelijk, met `if steps:` voor het deel dat verschilt, en voeg hem samen met `\"\".join(parts)`."],
        why="Het model gaat verder met het patroon van zijn prompt: voorbeelden met hun stappen vóór `The answer is` laten het stappen schrijven vóór zijn antwoord.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_read():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een programma dat het antwoord leest

        Maak `number(word)` en `read_answer(text)` af. `number` haalt de komma's uit een woord en de tekens `$ % . : ; ! ? * ( )` van de uiteinden, en geeft het woord als het dan alleen uit cijfers bestaat, anders `None`; `"1250".isdigit()` is `True`, `"19.50".isdigit()` is `False`. `read_answer` geeft het eerste getal na de eerste `The answer is`; zonder die woorden het laatste woord dat een getal is; `None` als er geen is. Een getal blijft tekst, zoals `"18"`, net als het goede antwoord in GSM8K. De check leest antwoorden uit de run van de course en teksten die ervoor gemaakt zijn.
        """),
        L.task("Maak `number` en `read_answer` af."),
    ])
    return


@app.cell(hide_code=True)
def exercise_read():
    box_read = L.code_box("read-answer", lines=18)
    box_read
    return (box_read,)


@app.cell(hide_code=True)
def check_read(box_read):
    result_read = L.check("read-answer", box_read.value)
    result_read
    return (result_read,)


@app.cell(hide_code=True)
def hints_read():
    L.hints(
        "read-answer",
        ["Met `The answer is`: de woorden van `text.split(\"The answer is\")[1]` op volgorde; zonder: de woorden van achteren af, `reversed(text.split())`. In beide het eerste woord waarvoor `number` niet `None` is."],
        why="De voorbeelden leren het model te eindigen met *The answer is* en het getal; een antwoord dat anders eindigt, heeft zijn uitkomst meestal als laatste getal.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_vote():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De stemming

        Maak `majority_vote(answers)` af: het antwoord dat het vaakst in de lijst staat, zonder `None`; van antwoorden die even vaak komen, het eerste; `None` als er geen is. Zo geeft `majority_vote(["18", "26", "18", None])` het antwoord `"18"`. De check probeert een paar groepen, en elke groep van 8 samples van de run van de course, zoals de checker ze las.
        """),
        L.task("Maak `majority_vote` af."),
    ])
    return


@app.cell(hide_code=True)
def exercise_vote():
    box_vote = L.code_box("majority-vote", lines=10)
    box_vote
    return (box_vote,)


@app.cell(hide_code=True)
def check_vote(box_vote):
    result_vote = L.check("majority-vote", box_vote.value)
    result_vote
    return (result_vote,)


@app.cell(hide_code=True)
def hints_vote():
    L.hints(
        "majority-vote",
        ["Tel in een dict, zoals in het uitgewerkte voorbeeld; loop dan de lijst op volgorde af en houd een antwoord alleen als het vaker komt dan het antwoord dat je had."],
        why="Wie de lijst op volgorde afloopt en alleen een grotere telling neemt, geeft een gelijkspel aan het antwoord dat het eerst kwam.",
    )
    return


@app.cell(hide_code=True)
def text_play_votes():
    mo.md(rf"""
    **Meer samples: stemmen en pass@k.** Voor elke vraag trok de course {C.SAMPLES} antwoorden met stappen bij temperature {C.number(C.TEMPERATURE, 1, 'nl')}. Het blok leest ze met je `read_answer` en stemt met je `majority_vote` (die van de course, tot beide goed zijn), over de eerste $k$ samples van elke vraag.
    """)
    return


@app.cell(hide_code=True)
def play_votes(box_read, box_vote, result_read, result_vote):
    _course_read, _course_vote = C.course_functions()
    _mine = result_read.state == "correct" and result_vote.state == "correct"
    _read = run_code(box_read.value).names["read_answer"] if _mine else _course_read
    _vote = run_code(box_vote.value).names["majority_vote"] if _mine else _course_vote
    _s = C.scores(R, _read, _vote)
    _labels = {"title": f"Goed van {_s['n']}, bij k samples", "x": "k, samples per vraag", "y": "goede antwoorden", "vote": "stemmen",
               "pass": "pass@k", "greedy": "greedy chain of thought", "lang": "nl",
               "said": "**k = {k}:** stemmen {vote}, pass@{k} {passes}, greedy {greedy} van {n}; ongeveer {tokens} tokens per vraag.",
               "alt": f"Goede antwoorden van {_s['n']} bij k van 1 tot {C.SAMPLES}: stemmen als doorgetrokken lijn, pass@k gestippeld, de greedy chain of thought als dunne vlakke lijn, en een gestippelde lijn bij de gekozen k."}
    _play = L.explore("votes", "k", lambda _v: C.votes_frame(_s, int(_v), _labels), start=1, low=1, high=C.SAMPLES, step=1,
              role="hyper", task="Maak k groter. Welke lijn blijft stijgen, en waarom kan stemmen niet mee?")
    _play
    return


@app.cell(hide_code=True)
def text_play_votes_after():
    mo.md(rf"""
    Pass@k gaat nooit omlaag als $k$ groeit, en eindigt op {S['pass'][-1]}: één goed antwoord is genoeg, als iets het nakijkt. Stemmen helpt alleen als het goede antwoord het vaakst voorkomt; met de checker van de course eindigt het op {S['vote'][-1]}, tegen {S['steps']} voor greedy, binnen de ruis van 30 vragen. Wang en anderen (2022) vonden 17,9 punten meer op GSM8K met veel grotere modellen; met één klein model en 30 vragen valt niet te zeggen of de winst bij 0,6B kleiner is of dat dit toeval is.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(rf"""
    ### Qwen3 met jouw prompt en jouw checker

    De demo geeft Qwen3-0.6B-Base de eerste {C.QUESTIONS['cpu']} testvragen met je `few_shot_prompt`, zonder en met stappen, greedy, en leest elk antwoord met je `read_answer`; op een GPU alle {C.QUESTIONS['gpu']}, met een stemming over {C.VOTE['gpu']} samples. Op de CPU van de course met 2 threads duurde het {C.number(C.DEMO['minutes'], 1, 'nl')} minuten, gemeten; op een GPU is het niet gemeten. Greedy schrijven ligt vast: op een CPU zoals die van de course ligt je lijn met stappen op die van de course, gestippeld; een ander apparaat kan door afronding een token veranderen.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B-Base (1,2 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    C.download_or_stop(get_button, {
        "ask": "*Qwen3-0.6B-Base staat nog niet op je computer. De knop downloadt het één keer van Hugging Face, `Qwen/Qwen3-0.6B-Base` op een vaste versie, 1,2 GB, onder de licentie Apache-2.0: dezelfde bestanden als in [Dezelfde stappen op een open model](/course/open/llm-same-on-open-model?lang=nl).*",
        "busy": "**Downloaden** {done} van {total} MB …",
        "no_net": "*Het downloaden lukte niet: Hugging Face was niet bereikbaar, of de verbinding viel weg. Kijk je verbinding na, en klik opnieuw.*",
        "not_pinned": "*Het downloaden lukte niet: het bestand dat binnenkwam is niet het vastgelegde bestand van de course, dus de course gooide het weg. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*Het downloaden lukte niet: er is op je schijf geen ruimte voor 1,2 GB. Maak ruimte, en klik opnieuw.*",
        "failed": "*Het downloaden lukte niet ({why}). Klik opnieuw.*",
    })
    mo.md("*Qwen3-0.6B-Base is gedownload.*")
    return


@app.cell(hide_code=True)
def run_button_cell():
    run_button = mo.ui.run_button(label="Laat Qwen3 je prompt beantwoorden")
    run_button
    return (run_button,)


@app.cell(hide_code=True)
def demo_run(box_prompt, box_read, box_vote, result_prompt, result_read, result_vote, run_button):
    ran = None
    if result_prompt.state != "correct" or result_read.state != "correct":
        _out = mo.md("*Maak eerst `few_shot_prompt` en `read_answer` goed in oefening 1 en 2: de demo gebruikt ze.*")
    elif not C.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif not run_button.value:
        _out = mo.md("*Druk op de knop om Qwen3 te laten antwoorden.*")
    else:
        _n = C.QUESTIONS[C.where()["kind"]]
        _live = L.live({"title": f"Tot nu toe goed, van de eerste {_n}", "x": "vraag", "y": "tot nu toe goed",
                        "alt": "Hoeveel antwoorden van je run tot nu toe goed zijn: met stappen als de lijn met punten, direct als grijze gestippelde lijn, en de run van de course met stappen op dezelfde vragen als een tweede gestippelde lijn."},
                       _n, role="ink", low=0, high=_n)
        _live.reference(C.so_far(R, "steps", _n), "", role="ink")
        _direct = []
        _what = {"direct": "direct", "steps": "met stappen", "votes": f"{C.VOTE['gpu']} samples"}

        def _show(_d):
            if _d["phase"] == "load":
                _live.line("**Qwen3 laden** …")
            elif _d["phase"] == "answer":
                _live.line(f"**Vraag {_d['i']} van {_d['n']}** · {_what[_d['kind']]} …")
            elif _d["phase"] == "answered":
                _live.line(f"**Vraag {_d['i']} van {_d['n']}** · {_what[_d['kind']]} · {_d['tokens']} tokens · {_d['seconds']:.0f} s · {'goed' if _d['right'] else 'fout'}")
            elif _d["phase"] == "question":
                _direct.append((_d["i"], _d["direct"]))
                _live.reference(_direct, "")
                _live.add(_d["i"], _d["steps"], f"**Vraag {_d['i']} van {_d['n']}:** tot nu toe {_d['steps']} goed met stappen, de lijn met punten, en {_d['direct']} direct, de grijze gestippelde lijn")

        _vote = run_code(box_vote.value).names["majority_vote"] if result_vote.state == "correct" else C.course_functions()[1]
        _r = C.run(run_code(box_prompt.value).names["few_shot_prompt"], run_code(box_read.value).names["read_answer"], _vote, _show)
        if isinstance(_r, str):
            _out = _live.stopped({
                "prompt": "*Je `few_shot_prompt` stopt met een fout bij een vraag van de demo, of geeft geen tekst. Haalt hij zijn check nog?*",
                "read": "*Je `read_answer` stopt met een fout bij een antwoord van Qwen3. Haalt hij zijn check nog?*",
                "vote": "*Je `majority_vote` stopt met een fout bij de samples van een vraag. Haalt hij zijn check nog?*",
                "run": "*Qwen3 laden stopt met een fout. Sluit andere programma's en druk opnieuw op de knop; gebeurt het weer, meld het dan op GitHub.*",
                "memory": "*Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop.*",
            }[_r])
        else:
            ran = _r
            _right = C.run_scores(_r)
            _out = mo.vstack([mo.Html(_live.picture()), mo.md(f"Goed: {_right['steps']} van {_right['n']} met stappen, de lijn met punten, en {_right['direct']} direct, de grijze gestippelde lijn. De run van de course met stappen is ook gestippeld: waar je hem niet ziet, ligt hij onder de jouwe.")])
    _out
    return (ran,)


@app.cell(hide_code=True)
def demo_result(ran):
    _c = C.run_scores(R)
    _m = C.run_scores(ran) if ran is not None else None
    _cols = [("de run van de course", _c)]
    if _m is not None:
        _cols = [(f"je run, {ran['device']}", _m)]
        if _m["n"] < _c["n"]:   # de antwoorden van de course op dezelfde vragen, naast al zijn eigen
            _cols.append((f"de run van de course, zijn eerste {_m['n']}", C.run_scores({"questions": R["questions"][:_m["n"]]})))
        _cols.append((f"de run van de course, alle {_c['n']}", _c))
    _runs = [_s for _n, _s in _cols]

    def _row(_name, _key):
        return f"| {_name} | " + " | ".join("" if _s[_key] is None else f"{_s[_key]} van {_s['n']}" for _s in _runs) + " |\n"

    def _two(_name, _key):
        return f"| {_name} | " + " | ".join(f"{C.number(_s[_key]['direct'], 1, 'nl')} en {C.number(_s[_key]['steps'], 1, 'nl')}" for _s in _runs) + " |\n"

    _head = "| | " + " | ".join(_n for _n, _s in _cols) + " |\n|---|" + "---|" * len(_cols) + "\n"
    mo.md(_head + _row("goed, direct", "direct") + _row("goed, met stappen", "steps") + _row(f"goed, stemming van {C.SAMPLES}", "vote")
          + _two("tokens per antwoord, direct en met stappen", "tokens") + _two("seconden per antwoord, direct en met stappen", "seconds")
          + "\nMet 10 vragen is één antwoord meer of minder 10 punten; met 30 is de standaardfout ongeveer 9 punten."
          + ("\n\n*De run van de course staat er, tot je Qwen3 zelf laat antwoorden.*" if _m is None else ""))
    return


@app.cell(hide_code=True)
def text_replay():
    mo.md(r"""
    **Jouw antwoorden, vraag voor vraag:** het blok van *Het idee*, nu over je run, met wat je `read_answer` las.
    """)
    return


@app.cell(hide_code=True)
def replay_cell(box_read, ran):
    if ran is None:
        _out = mo.md("*Als je Qwen3 hebt laten antwoorden, kun je hier door zijn antwoorden gaan.*")
    else:
        _read = run_code(box_read.value).names["read_answer"]
        _labels = {"title": "Vraag {i}: de tokens van elk antwoord", "x": "geschreven tokens", "direct": "direct", "steps": "met stappen",
                   "tokens": "tokens", "read": "gelezen:", "nothing": "niets", "key": "het goede antwoord is", "question": "Vraag {i} van {n}",
                   "alt": "De tokens van het directe antwoord en van het antwoord met stappen van je run op de gekozen vraag, als twee balken, elk met een vinkje als je read_answer het goede getal vond en een kruisje als dat niet zo is."}
        _out = L.explore("replay", "vraag", lambda _v: C.answer_frame(ran, int(_v), _read, _labels), start=1, low=1,
                         high=len(ran["questions"]), step=1, role="",
                         task="Zoek een antwoord dat je checker als niets leest. Wat schreef het model daar?")
    _out
    return


@app.cell(hide_code=True)
def text_replay_after():
    mo.md(rf"""
    In de run van de course leest de checker niets in de chains of thought van vraag {C.UNREAD[0]} en {C.UNREAD[1]}: ze eindigen bij een negatief getal, `-$10,000` en `-20`, en een woord met een minteken bestaat niet alleen uit cijfers. Geen van de goede antwoorden van deze {S['n']} vragen is negatief, dus hier verandert het geen score; een paar antwoorden van GSM8K zijn dat wel, en een checker daarvoor zou het teken moeten lezen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_faithful():
    mo.md(r"""
    ### Oefening 4. Is de chain de reden?
    """)
    return


@app.cell(hide_code=True)
def explain_faithful():
    explain_faithful_cell = L.explain(
        "faithful",
        question="Qwen3 schreef een chain of thought die eindigt bij het goede getal. Laat dat zien dat Qwen3 het getal vond *dankzij* die stappen? Hoe zou je dat testen?",
        model_answer="Nee. Een chain of thought kan een verhaal zijn dat bij het antwoord past, in plaats van de weg ernaartoe: als het goede antwoord in de voorbeelden van een prompt steeds (A) is, kiezen modellen vaker (A) zonder dat in hun stappen te zeggen (Turpin en anderen, 2023). Om het te testen verander je de chain en kijk je of het antwoord meegaat: zet een fout in een stap, of breek de chain af en vraag meteen om het antwoord (Lanham en anderen, 2023).",
        points=[
            "een goed antwoord met stappen laat niet zien dat de stappen het veroorzaakten",
            "de stappen kunnen bij het antwoord passen zonder het te veroorzaken, en een goed antwoord kan met foute stappen komen",
            "verander een stap, of breek de chain af, en kijk of het antwoord verandert",
            "dit heet faithfulness, en het is een open onderzoeksvraag",
        ],
    )
    explain_faithful_cell
    return (explain_faithful_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(rf"""
    ## Afronding

    **Samenhang.** [Hoe onderzoek werkt](/course/open/res-research-cycle?lang=nl) gebruikte het experiment van Kojima als voorbeeld; hier zie je waarom het werkt. Je checker is een programma zoals die van [Heeft post-training geholpen?](/course/open/llm-eval-chat?lang=nl), en DeepSeek-R1 is met zulke checkers getraind. In [Project: van nul een model dat zijn tussenstappen schrijft](/course/open/llm-reasoning-from-scratch?lang=nl) train je van nul een klein model dat zijn eigen tussenstappen schrijft, op data waarvan je elk voorbeeld zelf kiest.

    **Grenzen.** Redactiesommen zijn maar een klein deel van redeneren, en 30 vragen geven een standaardfout van ongeveer 9 punten. GSM8K kan in de pretraining van Qwen3 hebben gezeten (contaminatie), en dat kan de course niet nagaan. Een aanwijzing dat tekst in deze vorm erin zat: zonder enig voorbeeld schrijven {C.ZERO_STYLE['phrase']} van zijn {S['n']} zero-shot antwoorden toch *The answer is*. Namen en scores van modellen verouderen snel: deze zijn van oktober 2026; de eigen getallen van de les komen uit de run van de course.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_prompt, result_read, result_vote, explain_faithful_cell):
    L.wrap_up(result_prompt, result_read, result_vote, explain_faithful_cell)
    return


@app.cell(hide_code=True)
def look_back():
    _look = L.choice(
        "look-temperature",
        question=f"### Terugblik\n\nDe course trok zijn samples bij temperature {C.number(C.TEMPERATURE, 1, 'nl')}. Wat had temperature 0 gegeven?",
        options={
            "right": f"Hetzelfde antwoord, {C.SAMPLES} keer: het greedy antwoord",
            "wrong": f"{C.SAMPLES} verschillende antwoorden, allemaal fout",
            "faster": "Dezelfde antwoorden als bij 0,7, alleen sneller",
        },
        correct="right",
        why={
            "right": "Bij temperature 0 wint altijd het waarschijnlijkste token, dus elke sample is het greedy antwoord.",
            "wrong": "Bij temperature 0 is elke sample het greedy antwoord. Zie [Temperature: gedurfder of veiliger tekst](/course/open/lm-temperature?lang=nl).",
            "faster": "De temperature verandert welke tokens getrokken worden, niet hoe snel. Zie [Temperature: gedurfder of veiliger tekst](/course/open/lm-temperature?lang=nl).",
        },
    )
    _look
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
