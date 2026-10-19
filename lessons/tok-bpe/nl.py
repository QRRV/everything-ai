# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les tok-bpe. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw, play
    from everything_ai.lesson import lesson

    L = lesson("tok-bpe", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl) en [Dictionaries](/course/open/py-dicts?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rare():
    quick_rare = L.choice(
        "quick-rare",
        question="De tokenizer van GPT-2 krijgt een zeldzaam Nederlands woord dat niet als geheel in zijn vocabulary staat. Wat doet hij ermee?",
        options={
            "unknown": "Hij geeft het één special token dat betekent: onbekend woord",
            "skip": "Hij laat het woord weg",
            "right": "Hij knipt het in meerdere stukken die wel in zijn vocabulary staan",
        },
        correct="right",
        why={
            "unknown": "Zo'n token heeft GPT-2 niet: elke tekst wordt tokens uit zijn vocabulary. Zie [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl).",
            "skip": "Dan zou het model dat woord niet kunnen lezen of schrijven. Het valt in plaats daarvan uiteen in kleinere tokens. Zie [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl).",
            "right": "Vandaag leer je hoe zo'n vocabulary van stukken gemaakt wordt, en maak je er zelf een.",
        },
    )
    quick_rare
    return (quick_rare,)


@app.cell(hide_code=True)
def quick_check_key():
    quick_key = L.choice(
        "quick-key",
        question="Wat gebeurt er met deze regels?\n\n```python\ncounts = {}\ncounts[\"ab\"] = counts[\"ab\"] + 1\n```",
        options={
            "one": "`counts` wordt `{\"ab\": 1}`",
            "right": "Het stopt met een fout",
            "zero": "`counts` wordt `{\"ab\": 0}`",
        },
        correct="right",
        why={
            "one": "Een key die er niet is, kun je niet opzoeken, dus `counts[\"ab\"]` stopt voordat er iets wordt opgeteld. Zie [Dictionaries](/course/open/py-dicts?lang=nl).",
            "zero": "Een dictionary heeft geen waarde 0 voor keys die hij niet heeft: `\"ab\"` opzoeken stopt. Zie [Dictionaries](/course/open/py-dicts?lang=nl).",
            "right": "`\"ab\"` is nog geen key, en die opzoeken stopt het programma. Daarom begint een telling met `if pair in counts:`, zoals toen je paren van tekens telde; vandaag tel je paren van tokens.",
        },
    )
    quick_key
    return (quick_key,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        De vocabulary van een tokenizer wordt gemaakt uit tekst. Losse tekens zijn te klein: elk woord wordt veel tokens, dus een model ziet minder tekst tegelijk en moet elk woord uit zijn letters leren. Hele woorden zijn er te veel, en een woord dat niet in de lijst staat, zou helemaal niet gelezen of geschreven kunnen worden. **Byte pair encoding** zit ertussenin: begin met de tekens, en voeg steeds het paar dat het vaakst voorkomt samen tot één nieuw token. Vaak voorkomende woorden worden één token, zeldzame woorden blijven in stukken. GPT-2 maakte zijn vocabulary op deze manier. In deze projectstap bouw je het zelf: het eerste deel van je eigen tokenizer.
        """),
        draw.project_steps(["Letters en paren tellen", "Een bigram-model door te tellen", "Hetzelfde model, geleerd door een neural network",
                            "Je eigen tokenizer", "Je eigen GPT", "Een klein LLM pretrainen", "Het LLM instructies en voorkeuren laten volgen"],
                           3, here_label="← vandaag", title="Waar je project staat: je eigen LLM"),
    ])
    return


@app.cell(hide_code=True)
def predict_first():
    guess_first = L.guess(
        "first-merge",
        question="Je BPE gaat 100 merges leren uit de eerste 20.000 tekens van Tiny Shakespeare. Welk paar tekens voegt hij volgens jou als eerste samen?",
        options={
            "th": "`t` en `h`, zoals in the, that en thou",
            "ou": "`o` en `u`, zoals in you, your en thou",
            "right": "`e` en een spatie, het eind van veel woorden",
        },
        reveal="**`e` en een spatie**: 517 keer in die 20.000 tekens, vaker dan elk ander paar. Veel Engelse woorden eindigen op e, zoals the, be, me, are en have, en daarna komt een spatie. Na de eerste merge komen `t` en `h` het vaakst voor, 402 keer, en worden ze de tweede merge. BPE telt wat er staat, spaties ook.",
    )
    guess_first
    return (guess_first,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Byte pair encoding** (BPE) leert tokens uit een tekst. Begin met de tekst als lijst van tekens, `list(text)`: de eerste tokens. Tel elk paar naburige tokens, en voeg het paar dat het vaakst voorkomt samen tot één nieuw token, overal in de tekst. Dat paar is een **merge rule**. Tel dan opnieuw, met het nieuwe token tussen de andere, en voeg opnieuw samen: één keer voor elk nieuw token dat je wilt.

        Neem "aaabdaaabac". Het paar (`a`, `a`) komt 4 keer voor, vaker dan elk ander, dus het wordt `aa`: `aa a b d aa a b a c`. Nu komen (`aa`, `a`) en (`a`, `b`) allebei twee keer voor; neem het eerste, dus `aaa`. Daarna wordt (`aaa`, `b`) `aaab`: `aaab d aaab a c`, 5 tokens in plaats van 11. De regels bouwen op elkaar voort: `aaab` kwam uit drie merges.

        Om een nieuwe tekst te tokeniseren, begin je bij zijn tekens en pas je de merge rules toe in de volgorde waarin ze geleerd zijn. De tekens blijven in de vocabulary, dus elke tekst van bekende tekens kan getokeniseerd worden: een woord dat BPE nooit zag, blijft in kleinere stukken. GPT-2 leerde zo 50.000 merge rules. Een paar tokens is een tuple, zoals `("aa", "a")`, en een tuple kan een key van een dictionary zijn, net als een string. De twee strings aan elkaar plakken tot één key gaat niet: `"a"` en `"ab"` geven dezelfde `"aab"` als `"aa"` en `"b"`.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **byte pair encoding** (BPE) | tokens leren: begin met tekens, en voeg het vaakst voorkomende paar naburige tokens samen tot een nieuw token, keer op keer | `train_bpe(text, n)` |
        | **merge rule** | één geleerde stap van BPE: dit paar tokens wordt één token; een tokenizer past ze toe in de volgorde waarin ze geleerd zijn | `("t", "h")` |
        | een tuple als key | een paar tokens kan een key van een dictionary zijn, net als een string | `counts[("t", "h")]` |
        """),
        L.watch_out("Voeg samen van links naar rechts, en ga verder na beide tokens van een samengevoegd paar: `a a a` met de regel (`a`, `a`) geeft `aa a`, niet `aa aa`. In één ronde kan een token maar in één merge zitten."),
        mo.md(r"""
        De figuur: een regel uit Hamlet, getokeniseerd met de eerste $n$ merge rules die de course leerde uit de eerste 20.000 tekens van Tiny Shakespeare. ␣ is een spatie, ↵ een nieuwe regel. Schuif met de slider, of sleep over de grafiek, om $n$ te zetten: de grafiek toont hoeveel tokens de regel heeft bij elke $n$, en de zin eronder welk paar de nieuwste merge rule samenvoegt.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_rules_cell():
    def _frame(n):
        tokens = C.line_tokens(n)
        rules = "1 merge rule" if n == 1 else f"{n} merge rules"
        said = f"""Met {rules} heeft de versregel <span class="ea-num">{len(tokens)}</span> tokens."""
        if n:
            a, b = C.rule(n)
            said += f" Merge rule {n} voegt `{C.shown(a)}` en `{C.shown(b)}` samen tot `{C.shown(a + b)}`."
        return (C.line_frame(n, {"title": "Eén versregel, steeds verder samengevoegd", "row": rules + ": {k} tokens",
                                 "alt": f"To be, or not to be, that is the question, in {len(tokens)} tokens na {rules}: {' '.join(C.shown(t) for t in tokens)}.",
                                 "count_title": "Tokens in de versregel, bij elke n", "x": "merge rules", "y": "tokens",
                                 "count_alt": "Hoeveel tokens de regel heeft bij elk aantal merge rules van 0 tot 100: het daalt van 41 naar 20, het meeste in de eerste 40 merge rules."}),
                said)

    explore_rules = L.explore("merge-rules", "n", _frame, start=10, low=0, high=100, step=1, role="hyper", task="Bij welke merge rule wordt `the␣` één token, en uit welke twee tokens? Waarom wacht `be` tot merge rule 78, terwijl de versregel het twee keer heeft?")
    explore_rules
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Bij merge rule 19, uit `th` en `e␣`, de tokens van merge rule 2 en 1: de merge rules bouwen op elkaar voort. `be` wacht tot merge rule 78, omdat BPE leert van de eerste 20.000 tekens van Tiny Shakespeare, niet van deze versregel: daar komt `e␣` 517 keer voor en `b` `e` maar 63 keer, dus komen er 77 andere merges eerst. Ook na 100 merge rules is "question" nog niet één token: het eindigt als `q` `u` `es` `t` `i` `on`.

    ### Recept

    $n$ merge rules leren uit een tekst:

    1. De tekst als lijst van tekens: de eerste tokens.
    2. Tel elk paar naburige tokens; het vaakst voorkomende paar, het eerste bij gelijke stand, is de volgende merge rule.
    3. Voeg dat paar overal samen, van links naar rechts: de twee tokens worden één string, en het volgende paar begint erna.
    4. Stap 2 en 3 opnieuw, $n$ keer in totaal.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Eén ronde BPE op "aaabdaaabac". Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 3 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_round():
    # 1. de tekens: de eerste tokens
    _tokens = list("aaabdaaabac")
    # 2. tel elk paar buren, en neem het vaakst voorkomende, het eerste bij gelijke stand
    _counts = {}
    for _i in range(len(_tokens) - 1):
        _pair = (_tokens[_i], _tokens[_i + 1])
        if _pair in _counts:
            _counts[_pair] = _counts[_pair] + 1
        else:
            _counts[_pair] = 1
    print(_counts)
    _best = None
    _most = 0
    for _pair in _counts:
        if _counts[_pair] > _most:
            _best = _pair
            _most = _counts[_pair]
    print(_best)
    # 3. voeg het overal samen, van links naar rechts
    _out = []
    _i = 0
    while _i < len(_tokens):
        if _i < len(_tokens) - 1 and _tokens[_i] == _best[0] and _tokens[_i + 1] == _best[1]:
            # maak af: voeg _tokens[_i] + _tokens[_i + 1] toe, en ga verder met _i = _i + 2
            _out.append(_tokens[_i])
            _i = _i + 1
        else:
            _out.append(_tokens[_i])
            _i = _i + 1
    print(_out)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De tellingen zijn `{('a', 'a'): 4, ('a', 'b'): 2, ('b', 'd'): 1, ('d', 'a'): 1, ('b', 'a'): 1, ('a', 'c'): 1}`, dus het beste paar is `('a', 'a')`. Zoals de cel nu is, voegt de tak voor samenvoegen één token toe en gaat hij één verder, dus `_out` is weer de 11 tekens. Afgemaakt is het `['aa', 'a', 'b', 'd', 'aa', 'a', 'b', 'a', 'c']`: 9 tokens, en in `aaa` worden alleen de eerste twee samengevoegd.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_order():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. De volgorde van de regels
    """)
    return


@app.cell(hide_code=True)
def choice_order_cell():
    choice_order = L.choice(
        "order",
        question="Een BPE leerde twee merge rules: eerst (`h`, `e`), dan (`t`, `h`). In welke tokens knipt hij het woord \"the\"?",
        options={
            "right": "`t` en `he`: (`h`, `e`) gaat eerst",
            "the": "`the`: beide regels passen, dus het wordt één token",
            "th": "`th` en `e`: de regel (`t`, `h`) maakt `th`, en `e` blijft alleen",
        },
        correct="right",
        why={
            "the": "Om `the` te worden, zou hij een regel (`t`, `he`) of (`th`, `e`) nodig hebben, en die heeft hij geen van beide. Na (`h`, `e`) is er geen `h`-token meer over voor (`t`, `h`).",
            "th": "De regels gaan in de volgorde waarin ze geleerd zijn. (`h`, `e`) werd eerst geleerd, dus gaat eerst, en neemt de `h`.",
            "right": "(`h`, `e`) maakt `he`, en daarna zijn `t` en `h` geen buren meer. Een tokenizer past zijn regels elke keer toe in de volgorde waarin hij ze leerde, dus dezelfde tekst krijgt altijd dezelfde tokens.",
        },
    )
    choice_order
    return (choice_order,)


@app.cell(hide_code=True)
def text_exercise_merge():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een paar samenvoegen

        Maak `merge(tokens, pair)` af: de functie geeft een nieuwe lijst tokens waarin elke plek waar `pair[1]` op `pair[0]` volgt één token is, de twee strings aan elkaar: stap 3 van het recept. De loop en de `if` zijn gegeven. De check roept `merge(["l", "o", "w", " ", "l", "o", "w", "e", "r"], ("l", "o"))` aan, wat `["lo", "w", " ", "lo", "w", "e", "r"]` moet geven, en probeert andere paren. Als hij goed is, bewaart de check hem in je project, als `work/projects/llm/merge.py`.
        """),
        L.task("Laat `merge` van elke plek van `pair` één token maken."),
    ])
    return


@app.cell(hide_code=True)
def exercise_merge():
    box_merge = L.code_box("merge", lines=13)
    box_merge
    return (box_merge,)


@app.cell(hide_code=True)
def check_merge(box_merge):
    result_merge = L.check("merge", box_merge.value)
    result_merge
    return (result_merge,)


@app.cell(hide_code=True)
def hints_merge():
    L.hints(
        "merge",
        [
            "Onder de `if` worden twee tokens één: zet de twee strings aan elkaar met `+`, voeg dat toe aan `out`, en sla ze allebei over. Onder `else` gaat het token naar `out` zoals het is.",
            "Stap 3 van het afgemaakte uitgewerkte voorbeeld, met `tokens`, `pair` en `i` in plaats van `_tokens`, `_best` en `_i`.",
        ],
        why="Doordat je na beide tokens verdergaat, overlappen merges nooit: in `[\"a\", \"a\", \"a\"]` worden de eerste twee `\"aa\"`, en het derde blijft alleen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_train(result_merge):
    result_merge  # the note changes when exercise 2 saves your merge
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De merge rules leren

        Maak `train_bpe(text, n)` af: de functie leert `n` merge rules uit `text` en geeft ze als lijst van paren, in de volgorde waarin hij ze leerde: het recept. `pair_counts` en de loop zijn gegeven; de eerste regel van het vak importeert je `merge` uit je project. De check roept `train_bpe("low lower lowest", 3)` aan, wat `[("l", "o"), ("lo", "w"), (" ", "low")]` moet geven, en probeert andere teksten. Als hij goed is, bewaart de check hem als `work/projects/llm/bpe.py`.
        """),
        L.course_versions(),
        L.task("Laat `train_bpe` `n` merge rules leren, elk uit de tokens van de ronde ervoor."),
    ])
    return


@app.cell(hide_code=True)
def exercise_train():
    box_train = L.code_box("train-bpe", lines=26)
    box_train
    return (box_train,)


@app.cell(hide_code=True)
def check_train(box_train, result_merge):
    result_merge  # check again when exercise 2 saves your merge
    result_train = L.check("train-bpe", box_train.value)
    result_train
    return (result_train,)


@app.cell(hide_code=True)
def hints_train():
    L.hints(
        "train-bpe",
        [
            "`best` vinden is stap 2 van het uitgewerkte voorbeeld, met `counts`: begin met `best = None` en een hoogste telling van 0, en laat een paar ze alleen overnemen als zijn telling hoger is.",
            "Voeg na `merges.append(best)` het paar samen met je `merge`, en bewaar wat die geeft in `tokens`, voor de volgende ronde.",
        ],
        why="Elke ronde telt de paren van de tokens van de ronde ervoor, dus een regel kan voortbouwen op een eerdere: (`l`, `o`) en dan (`lo`, `w`) maken van \"low\" één token.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    De cel hieronder gebruikt je project. Hij traint je BPE op de eerste 20.000 tekens van Tiny Shakespeare, 100 merge rules, en tokeniseert dan met je `merge`: wat je typt, en de volgende 5.000 tekens, waar hij niet van leerde. Het werkt zodra oefening 2 en 3 af zijn, en duurt ongeveer een seconde.
    """)
    return


@app.cell(hide_code=True)
def demo_bpe(result_merge, result_train):
    result_merge, result_train  # run again when exercise 2 or 3 saves your module
    _result = C.demo()
    if _result is None:
        _out = mo.md("*Maak eerst oefening 2 en 3 af: dan staan je `merge` en je `train_bpe` in je project, en gebruikt deze cel ze.*")
    elif isinstance(_result, str):
        _out = mo.md({
            "slow": "*Je BPE duurt hier te lang: meer dan een minuut. Doet je `train_bpe` of je `merge` meer werk dan nodig is, zoals een loop die te vaak draait?*",
            "bpe": "*`merge` of `train_bpe` in `work/projects/llm/` stopt met een fout. Heb je het bestand veranderd? Verwijder het, en check oefening 2 of 3 opnieuw: de check bewaart je code dan opnieuw.*",
        }[_result])
    else:
        _rules = ", ".join(f"`{C.shown(a)}` + `{C.shown(b)}`" for a, b in _result["merges"][:10])
        _who = "" if not _result["stand_ins"] else " De versie van de course sprong in voor " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", zoals de melding bij oefening 3 zegt."
        _out = mo.vstack([
            mo.md(f"""Je eerste 10 merge rules: {_rules}. ␣ is een spatie, ↵ een nieuwe regel. Typ hieronder een tekst: je BPE tokeniseert hem terwijl je typt."""),
            mo.ui.anywidget(play.Tokens(*C.tokenizer(_result), text=C.LINE, examples=[C.LINE, "First Citizen:\nWe are accounted poor citizens", "Long live the king!"],
                                        decimal="," if L.lang == "nl" else ".", labels={"task_kick": L.w["task"], "task": "Typ <code>Citizen</code> en <code>king</code>: welke van de twee maakt je tokenizer één token? Waarom die?",
                                                "box": "Een tekst om te tokeniseren", "again": "Nog een voorbeeld", "chars": "tekens", "tokens": "tokens",
                                                "per": "tekens per token", "token": "Token", "bytes": "bytes", "char": "teken", "one": "token", "byte": "byte"})),
            mo.md(f"""`Citizen` is één token, en `king` twee: `k` en `ing`. De 20.000 tekens waar je BPE van leerde, zijn het begin van een toneelstuk, *Coriolanus*, waarin alleen "First Citizen:" al 23 keer voorkomt. BPE leert wat vaak voorkomt in de tekst waar hij van leert, niet in het Engels als geheel.

Op de 5.000 tekens daarna, waar hij niet van leerde, bevat een token gemiddeld {C.number(_result['per_token'], 2, L.lang)} tekens. Veel tokens eindigen op een spatie, zoals `to␣`: hier plakt de spatie aan het woord ervoor. GPT-2 knipt een tekst eerst in woorden, elk met de spatie ervoor, en voegt alleen binnen die woorden samen; daarom beginnen zijn tokens met een spatie, zoals `␣the`.{_who}"""),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Een bigram-model door te tellen](/course/open/lm-bigram-counts?lang=nl) telde je paren van tekens; BPE telt paren van tokens, en maakt van de vaakst voorkomende nieuwe tokens. Zo is de vocabulary uit [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl) gemaakt, uit veel meer tekst. Je project `llm` heeft nu `merge.py` en `bpe.py`. Hierna: *Byte-level BPE*, hetzelfde op de bytes van een tekst in plaats van zijn tekens, zodat elke tekst, in elke taal of met emoji, getokeniseerd kan worden.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_order, result_merge, result_train):
    L.wrap_up(choice_order, result_merge, result_train)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
