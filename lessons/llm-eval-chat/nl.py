# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-eval-chat. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-eval-chat", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Project: DPO, voorkeuren zonder reward model](/course/open/llm-dpo?lang=nl) en [Testdata uit de training data houden](/course/open/llm-decontamination?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_checker():
    quick_checker = L.choice(
        "quick-checker",
        question="Wat besliste in het project over DPO welk antwoord van een paar gekozen werd, en wat telde daarna de testinstructies?",
        options={
            "people": "Mensen kozen de antwoorden; `words_used` telde de testinstructies",
            "right": "Beide keren `words_used`: die maakte de paren, en die telde",
            "reward": "Een reward model koos de antwoorden; mensen telden de testinstructies",
        },
        correct="right",
        why={
            "people": "Er deden geen mensen mee: de checker maakte elk paar. Zie [Project: DPO, voorkeuren zonder reward model](/course/open/llm-dpo?lang=nl).",
            "reward": "DPO heeft geen reward model nodig: de paren kwamen van de checker. Zie [Project: DPO, voorkeuren zonder reward model](/course/open/llm-dpo?lang=nl).",
            "right": "Onthoud dat: het model leerde wat die checker beloont, en dezelfde checker gaf het zijn score.",
        },
    )
    quick_checker
    return (quick_checker,)


@app.cell(hide_code=True)
def quick_check_clean():
    quick_clean = L.choice(
        "quick-clean",
        question="De helft van de vragen van een benchmark blijkt in de training data te staan. Welke score zegt nog iets over wat het model kan?",
        options={
            "all": "De score op alle vragen: het model moest ze toch beantwoorden",
            "half": "De score op alle vragen, gehalveerd",
            "right": "De score op de schone subset: de vragen die niet in de training data stonden",
        },
        correct="right",
        why={
            "all": "Op de vragen die het zag, kan een model uit het geheugen antwoorden: die score mengt wat het onthield met wat het kan. Zie [Testdata uit de training data houden](/course/open/llm-decontamination?lang=nl).",
            "half": "Halveren lost niets op: de geziene vragen kunnen uit het geheugen beantwoord worden, en ze zijn niet zomaar de helft waard. Zie [Testdata uit de training data houden](/course/open/llm-decontamination?lang=nl).",
            "right": "Vandaag maak je die subset voor instructies, waar niet de tekst held out moet zijn, maar de gevraagde woorden.",
        },
    )
    quick_clean
    return (quick_clean,)


@app.cell(hide_code=True)
def text_why():
    _c = C.MEASURED["course"]
    mo.md(f"""
    ## Waarom

    SFT en DPO brachten het model van de course van {_c['base']['with']} naar {_c['sft']['with']} naar {_c['dpo']['with']} van de {_c['total']['with']} testinstructies uit [Project: leer je model instructies volgen (SFT)](/course/open/llm-sft?lang=nl). Heeft post-training geholpen? Twee twijfels. De checker die die instructies telt, is dezelfde als die de DPO-paren maakte, dus DPO kan die checker geleerd hebben in plaats van instructies. En de meeste testinstructies waren ook trainingsinstructies: om dezelfde woorden werden de modellen tijdens hun training gevraagd. Een eerlijkere test vraagt andere dingen die een programma kan controleren, op prompts waarop het model nooit trainde. Voor grote modellen test IFEval (Zhou en anderen, 2023) instructies volgen met zulke checks: 25 soorten verifieerbare instructies, zoals *gebruik deze trefwoorden*, *gebruik deze woorden niet* en *antwoord in hooguit zoveel zinnen*, in 541 prompts.

    **Een voorbeeld uit het dagelijks leven.** Een leerling-automobilist die één route geoefend heeft, slaagt voor een examen op die route. Om te weten of die kan rijden, test je op een andere route, en vraag je om een afslag die nooit geoefend is.
    """)
    return


@app.cell(hide_code=True)
def predict_without():
    guess_without = L.guess(
        "without",
        question="SFT en DPO leerden het model van de course verhalen schrijven die de gevraagde woorden gebruiken. Vraag het nu om een verhaal *zonder* het woord *dog*. Laat het model na DPO het woord vaker weg dan het model voor post-training, ongeveer even vaak, of minder vaak?",
        options={
            "more": "Vaker: het heeft instructies leren volgen",
            "same": "Ongeveer even vaak: het trainde nooit op deze soort",
            "right": "Minder vaak: het heeft geleerd het woord te gebruiken dat het krijgt",
        },
        reveal=f"**Minder vaak.** Van {C.WITHOUT} zulke prompts liet het model van de course het woord {C.MEASURED['course']['base']['without']} keer weg voor post-training, {C.MEASURED['course']['sft']['without']} keer na SFT, en {C.MEASURED['course']['dpo']['without']} keer na DPO. Het leerde nooit een instructie lezen: het leerde dat een woord in de instructie in het verhaal hoort.",
    )
    guess_without
    return (guess_without,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een verifieerbare instructie** is er een waarvan een kort programma het resultaat kan controleren: *gebruik het woord kite*, *gebruik het woord dog niet*, *schrijf hooguit drie zinnen*. Geen mens en geen model hoeft te oordelen, dus de check is voor elk antwoord en elk model dezelfde. Hij is niet perfect: een verhaal kan *kite* gebruiken zonder goed te zijn, en `sentences` telt *Mr.* als einde van een zin. Maar een vaste check die een beetje fout zit, zit voor elk model op dezelfde manier fout, en dat is wat een vergelijking nodig heeft.

        **Meer dan één soort.** Een model dat op één soort instructie is gefinetuned, kan daarin beter worden en in andere slechter. Daarom vraagt de test drie soorten: die waarop de modellen trainden, *met de woorden*, en twee die ze nooit zagen, *zonder een woord* en *hooguit een aantal zinnen*.

        **Held out door wat varieert.** Alle prompts van een soort delen hun sjabloon: *Write a story with the words: … Story:*. Een check op n-grams, zoals bij decontaminatie, kan ze dus niet uit elkaar houden: met een kleine n deelt elke prompt de n-grams van zijn sjabloon duizenden keren met de training data, en met n = 13 is een prompt van acht tot tien woorden korter dan één n-gram, dus deelt hij er geen. Wat held out moet zijn, is het deel dat varieert: de gevraagde woorden. Een prompt waarvan de woorden, in welke volgorde ook, nooit een trainingsinstructie waren, hoort bij de **schone subset**.
        """),
        draw.flow(
            [("Prompt", ["Write a story without", "the word: dog."]),
             ("Model", ["basis, SFT", "of SFT en DPO"]),
             ("Check", ["avoids(story, \"dog\")", "True of False"])],
            forward=["antwoord", "verhaal"],
            back=[],
            title="Een verifieerbare instructie"),
    ])
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.md(r"""
    ### Nieuwe woorden en tekens

    | Woord of teken | Wat het betekent | In code |
    |---|---|---|
    | **verifieerbare instructie** | een instructie waarvan een kort programma het resultaat kan controleren, elke keer op dezelfde manier | `avoids(story, word)` |
    | **verifieerbare eval** | een test van instructies volgen, gemaakt van verifieerbare instructies, op held-out prompts, voor elk model dezelfde | |

    ### Recept

    Om te testen of post-training geholpen heeft:

    1. Kies instructies die een programma kan controleren, van meer dan één soort: ook soorten waarop het model nooit trainde.
    2. Maak de prompts held out: het deel dat varieert mag nooit een trainingsinstructie geweest zijn.
    3. Leg elk model dezelfde prompts voor, op dezelfde manier getrokken, met dezelfde seed per prompt.
    4. Tel per soort hoeveel antwoorden slagen; en kijk of slagen afhangt van wat er gevraagd werd.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De eerste prompt van de soort *zonder*, en de antwoorden van de drie modellen van de course erop, elk gecontroleerd.
    """)
    return


@app.cell
def example_check():
    _d = C.course()
    _k = 100   # de eerste prompt van de soort "without"
    _kind, _words, _prompt = _d["prompts"][_k]
    print(_prompt.split("\n")[0])
    for _m in C.MODELS:
        _story = _d["answers"][_m][_k]
        _found = {_w.strip(".,!?;:\"'") for _w in _story.lower().split()}
        print(_m, _words[0] not in _found, _story[:60] + " ...")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `True` betekent dat het verhaal het woord weglaat, zoals de prompt vraagt. De check is het omgekeerde van `words_used`, woord voor woord: dezelfde woorden van het verhaal, met dezelfde leestekens eraf.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_avoids():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Zonder het woord

        Maak `avoids(story, word)` af. Geef `True` terug als het verhaal `word` niet gebruikt, en `False` als het dat wel doet, en zie de woorden van het verhaal zoals `words_used` ze ziet: in kleine letters, zonder de leestekens eromheen. De check roept `avoids("The kite flew over the park.", "dog")` aan, en andere.
        """),
        L.task("Laat `avoids` zeggen of een verhaal een woord weglaat."),
    ])
    return


@app.cell(hide_code=True)
def exercise_avoids():
    box_avoids = L.code_box("avoids", lines=6)
    box_avoids
    return (box_avoids,)


@app.cell(hide_code=True)
def check_avoids(box_avoids):
    result_avoids = L.check("avoids", box_avoids.value)
    result_avoids
    return (result_avoids,)


@app.cell(hide_code=True)
def hints_avoids():
    L.hints(
        "avoids",
        [
            "Je `words_used` uit de les over SFT ziet de woorden van een verhaal al zoals deze check moet. Het verhaal staat hier al in kleine letters: wat deed die verder met elk woord?",
            "Zodra één woord van het verhaal het woord is dat weg moet, ligt het antwoord vast. En als de loop elk woord gezien heeft?",
        ],
        why="De twee checks zien woorden precies op dezelfde manier, dus een woord dat `words_used` zou vinden, is er een waarop `avoids` het verhaal afkeurt.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_sentences():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Zinnen tellen

        Maak `sentences(story)` af. Geef terug hoeveel zinnen het verhaal heeft, met deze regel: een zin eindigt bij een `.`, `!` of `?`, en een reeks ervan, zoals `...` of `?!`, sluit één zin af, niet meer. De check roept `sentences("Tom ran. He fell... Was he hurt?")` aan, en andere.
        """),
        L.task("Laat `sentences` de reeksen `.`, `!` en `?` tellen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_sentences():
    box_sentences = L.code_box("sentences", lines=10)
    box_sentences
    return (box_sentences,)


@app.cell(hide_code=True)
def check_sentences(box_sentences):
    result_sentences = L.check("sentences", box_sentences.value)
    result_sentences
    return (result_sentences,)


@app.cell(hide_code=True)
def hints_sentences():
    L.hints(
        "sentences",
        [
            "In `...` begint alleen de eerste punt iets nieuws. Wat geldt voor het teken voor die eerste punt, en niet voor de tekens voor de andere twee?",
            "`before` onthoudt of het vorige teken een zin afsloot. Wanneer telt een teken één op bij `count`?",
        ],
        why="Een reeks eindtekens is één einde: alleen tellen waar een reeks begint geeft er één per zin, of die nu eindigt op `.`, `?!` of `...`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_held():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De schone subset

        Maak `held_out(prompts, train)` af. Het zijn allebei lijsten van gevraagde woorden, één lijst per instructie: `prompts` die van de test, `train` die waarop de modellen trainden. Geef de prompts terug, in hun volgorde, waarvan de woorden, in welke volgorde ook, nooit een trainingsinstructie waren. De check roept `held_out([["park", "kite"], ["dog"], ["sun", "park"]], [["kite", "park"], ["cat"]])` aan, en andere.
        """),
        L.task("Laat `held_out` de prompts houden waarvan de woorden samen nooit een trainingsinstructie waren."),
    ])
    return


@app.cell(hide_code=True)
def exercise_held():
    box_held = L.code_box("held-out", lines=8)
    box_held
    return (box_held,)


@app.cell(hide_code=True)
def check_held(box_held):
    result_held = L.check("held-out", box_held.value)
    result_held
    return (result_held,)


@app.cell(hide_code=True)
def hints_held():
    L.hints(
        "held-out",
        [
            "*kite and park* en *park and kite* zijn dezelfde instructie. Hoe maak je van allebei hetzelfde, iets wat een set kan bevatten?",
            "Een prompt blijft als zijn woorden, op dezelfde manier gemaakt, niet bij die van de trainingsinstructies staan. Valt hij af als hij maar één woord deelt?",
        ],
        why="Het sjabloon is voor elke prompt hetzelfde, dus de woorden beslissen of een prompt nieuw is; gesorteerd doet de volgorde er niet meer toe.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.vstack([
        mo.md(f"""
        ### Test de drie modellen

        Het basismodel van de course uit [Capstone: je eigen kleine LLM pretrainen](/course/open/llm-pretrain-capstone?lang=nl), zijn SFT-model, en dat model met zijn DPO-adapters beantwoordden alle {len(C.prompts())} prompts: de {C.T.TESTS} testinstructies van de les over SFT, {C.WITHOUT} om een verhaal zonder een woord, en {C.SENTENCES} om hooguit 1 tot 5 zinnen. Je checks scoren hun antwoorden hieronder; de schone subset is wat je `held_out` overhoudt van de testinstructies, tegen de instructies van de SFT-run van de course (voor je eigen modellen: van jouw SFT-run) en van de DPO-paren. Heb je eigen modellen, dan laat de knop ze dezelfde prompts beantwoorden: op de CPU van de course, met één thread, duurde dat ongeveer {round(C.MEASURED['yours']['seconds'] / 60)} minuten voor haar modellen van de demoruns.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def button_cell():
    _have = [_m for _m, _there in C.mine().items() if _there]
    ask_button = mo.ui.run_button(label="Laat mijn modellen antwoorden", disabled=not _have)
    ask_button
    return (ask_button,)


@app.cell(hide_code=True)
def demo_ask(ask_button):
    asked = None
    _note = None
    if ask_button.value:
        def _progress(_p):
            mo.output.replace(mo.md(f"**{_p['model']}** · prompt {_p['done']} van {_p['total']} · {C.number(_p['seconds'] / 60, 1, L.lang)} minuten"))
        _r = C.ask(_progress)
        if _r == "run":
            _note = "*Je modellen kunnen niet antwoorden: een module van je project stopt met een fout. Slagen je modules in `work/projects/llm/` nog voor hun checks?*"
        else:
            asked = _r
    mo.output.replace(mo.md(_note)) if _note else mo.output.clear()
    return (asked,)


@app.cell(hide_code=True)
def demo_score(asked, box_avoids, box_held, box_sentences, result_avoids, result_held, result_sentences):
    _course = {"avoids": run_code(C.EXERCISES["avoids"].solution).names["avoids"],
               "sentences": run_code(C.EXERCISES["sentences"].solution).names["sentences"],
               "held_out": run_code(C.EXERCISES["held-out"].solution).names["held_out"]}
    _boxes = {"avoids": (box_avoids, result_avoids), "sentences": (box_sentences, result_sentences), "held_out": (box_held, result_held)}
    _fns, _notes = {}, []
    for _k, (_box, _result) in _boxes.items():
        _fns[_k] = run_code(_box.value).names.get(_k) if _result.state == "correct" else None
        if _fns[_k] is None:
            _fns[_k] = _course[_k]
            _notes.append(f"*De `{_k}` van de course staat er tot de jouwe slaagt voor zijn check.*")
    _t = C.tools()
    _wu = run_code(C.T.EXERCISES["words-used"].solution).names["words_used"] if _t == "run" else _t["words_used"]
    if asked is not None and asked["answers"]:
        _answers, _whose = asked["answers"], "je modellen"
        _train = asked["train"]
    else:
        _answers, _whose = C.course()["answers"], "de modellen van de course"
        _train = C.trained({"course"})
    _s = C.score(_answers, _train, _fns["avoids"], _fns["sentences"], _fns["held_out"], _wu)
    if isinstance(_s, str):   # one of the checks stops on the answers: the course's four stand in
        _notes.append(f"*Je `{_s if _s != 'words' else 'words_used'}` stopt met een fout op de antwoorden, of geeft geen antwoord van de goede soort: de checks van de course staan er.*")
        _s = C.score(_answers, _train, _course["avoids"], _course["sentences"], _course["held_out"],
                     run_code(C.T.EXERCISES["words-used"].solution).names["words_used"])
    _names = {"base": "voor post-training", "sft": "na SFT", "dpo": "na SFT en DPO"}
    _tot = _s["total"]
    _rows = [f"| {_names[_m]} | {_s[_m]['with']} van {_tot['with']} | {_s[_m]['clean']} van {_tot['clean']} | {_s[_m]['without']} van {_tot['without']} | {_s[_m]['sentences']} van {_tot['sentences']} |"
             for _m in C.MODELS if _m in _s]
    _per = [f"| {_names[_m]} | " + " | ".join(C.number(_s[_m]['per'].get(_n, 0), 1, L.lang) for _n in range(1, 6)) + " |" for _m in C.MODELS if _m in _s]
    mo.vstack([
        mo.md(f"Hoeveel antwoorden van {_whose} slagen, per soort:\n\n"
              "| | met de woorden, alle | met de woorden, schone subset | zonder het woord | hooguit N zinnen |\n|---|---|---|---|---|\n" + "\n".join(_rows)
              + "\n\nHet gemiddelde aantal zinnen, voor elk aantal N waar de prompt om vroeg:\n\n"
              "| | N = 1 | N = 2 | N = 3 | N = 4 | N = 5 |\n|---|---|---|---|---|---|\n" + "\n".join(_per)),
        *[mo.md(_n) for _n in _notes],
    ])
    return


@app.cell(hide_code=True)
def text_after_demo():
    _c, _y = C.MEASURED["course"], C.MEASURED["yours"]
    mo.md(f"""
    Op *met de woorden* hielp post-training veel: voor de modellen van de course van {_c['base']['with']} naar {_c['dpo']['with']} van de {_c['total']['with']}, voor haar modellen van de demoruns van {_y['base']['with']} naar {_y['dpo']['with']}. Maar de meeste van die instructies waren ook trainingsinstructies, en op de schone subset verschillen de twee: het model van de course na DPO deed er {_c['dpo']['clean']} van zijn {_c['total']['clean']}, haar model van de demoruns {_y['dpo']['clean']} van {_y['total']['clean']}. Alleen de schone subset zegt of wat een model leerde meegaat naar woorden die het nooit samen moest gebruiken. Op *zonder het woord* maakte DPO beide slechter dan ze na SFT waren: van {_c['sft']['without']} naar {_c['dpo']['without']}, en van {_y['sft']['without']} naar {_y['dpo']['without']}, van de {_c['total']['without']}. En op *hooguit N zinnen* schrijven de modellen na SFT en DPO ongeveer evenveel zinnen wat N ook vraagt: ze lezen het getal niet, dus wat slaagt hangt af van hoe lang hun verhalen toevallig zijn. Post-training leerde de soort instructie waarop het trainde, en zijn checker; niet instructies volgen in het algemeen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ### Oefening 4. Waarom niet?
    """)
    return


@app.cell(hide_code=True)
def choice_why_cell():
    choice_why = L.choice(
        "why-not",
        question="Na DPO deed het model van de course de meeste testinstructies van de les over SFT, tegen een kwart na SFT. Waarom laat dat niet zien dat DPO het instructies leerde volgen?",
        options={
            "right": "De meeste testinstructies waren trainingsinstructies, en hun checker maakte de DPO-paren",
            "small": "Honderd instructies zijn te weinig om iets over een model te zeggen",
            "base": "Het basismodel deed er bijna geen, dus elk model zou er beter uitzien",
        },
        correct="right",
        why={
            "small": "Honderd is weinig, maar het verschil is groot. Het probleem is wat er gemeten werd: instructies waarop het model trainde, geteld door de checker waartegen het trainde.",
            "base": "De vergelijking is met het model na SFT, niet alleen met het basismodel. Het probleem is wat er gemeten werd: instructies waarop het model trainde, geteld door de checker waartegen het trainde.",
            "right": "Op held-out prompts, en op een instructie waarop het nooit trainde, doet het model na DPO het veel slechter.",
        },
    )
    choice_why
    return (choice_why,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De checker `words_used` uit [Project: leer je model instructies volgen (SFT)](/course/open/llm-sft?lang=nl), nu een van drie soorten; de schone subset uit [Testdata uit de training data houden](/course/open/llm-decontamination?lang=nl), nu voor de woorden van een instructie; en de modellen uit [Project: DPO, voorkeuren zonder reward model](/course/open/llm-dpo?lang=nl), nu naast elkaar gezet. Hierna: testen of een model weigert wat het moet weigeren, en alleen dat.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_avoids, result_sentences, result_held, choice_why):
    L.wrap_up(result_avoids, result_sentences, result_held, choice_why)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
