# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson repro-guided-build. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw, play
    from everything_ai.lesson import lesson

    L = lesson("repro-guided-build", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Begeleide reproductie 1: lees de paper en kies de claim](/course/open/repro-guided-claim?lang=nl) en [Project: je GPT een modern LLM maken](/course/open/llm-modern-gpt?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_verdict():
    quick_verdict = L.choice(
        "quick-verdict",
        question="Je plan zegt dat een run grokte als de validatie 99% haalt op een stap die minstens 10 keer zo laat is als bij de training. Een run heeft t = 400 en v = 3.000. Wat is zijn oordeel?",
        options={
            "grokked": "\"grokked\": de validatie kwam er veel later dan de training",
            "right": "\"no delay\": 3.000 is minder dan 10 keer 400",
            "not": "\"not reached\": de validatie heeft meer stappen nodig",
        },
        correct="right",
        why={
            "grokked": "Later, maar volgens het plan niet laat genoeg: 10 keer 400 is 4.000, en 3.000 is minder. Zie [Begeleide reproductie 1](/course/open/repro-guided-claim?lang=nl).",
            "not": "De validatie haalde 99% wel, op stap 3.000. Zie [Begeleide reproductie 1](/course/open/repro-guided-claim?lang=nl).",
            "right": "Het plan beslist, niet hoe de curves eruitzien: daarom is het vóór elke run geschreven.",
        },
    )
    quick_verdict
    return (quick_verdict,)


@app.cell(hide_code=True)
def quick_check_kv():
    quick_kv = L.choice(
        "quick-kv",
        question="De transformer van de paper heeft 4 heads, elk met eigen keys en values. Je ModernGPT maak je als `ModernGPT(vocab, d, h, kv_heads, layers)`. Welke `kv_heads` geeft met `h = 4` de attention van de paper?",
        options={
            "one": "1: één key head en één value head, gedeeld door alle 4 query heads",
            "right": "4: evenveel key en value heads als query heads",
            "two": "2: elke key en value head gedeeld door 2 query heads, zoals in de GPT van je project",
        },
        correct="right",
        why={
            "one": "Dat is multi-query attention: die houdt minder in het geheugen, maar het is niet de transformer van de paper. Zie [Grouped-query attention](/course/open/llm-gqa?lang=nl).",
            "two": "Dat is grouped-query attention, een keuze van de GPT van je project, niet van de paper. Zie [Project: je GPT een modern LLM maken](/course/open/llm-modern-gpt?lang=nl).",
            "right": "Dan heeft elke query head eigen keys en values, zoals in de paper; de demo hieronder bouwt je ModernGPT zo.",
        },
    )
    quick_kv
    return (quick_kv,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Een reproductie toetst de claim van de paper alleen als hij het experiment van de paper draait: dezelfde data, hetzelfde model en dezelfde instellingen. Elke instelling die je uit je hoofd invult of van een standaardwaarde overneemt, is een tweede experiment dat in het eerste verstopt zit. Daarom bouw je in deze stap de data precies zoals de paper hem maakt, haal je elke instelling van de plek in de paper die hem geeft, en schrijf je op waar je model afwijkt. Het model is je eigen ModernGPT, aangepast; de data en de instellingen gaan in je reproductiekit.
        """),
        draw.project_steps(["Lees de paper en kies de claim", "Bouw het model en de data", "Sanity checks", "Reproduceer de belangrijkste figuur",
                            "De ablation", "Vergelijk met de paper", "Het verslag"],
                           1, here_label="← vandaag", title="Waar je project staat: je reproductie"),
    ])
    return


@app.cell(hide_code=True)
def predict_size():
    guess_size = L.guess(
        "size",
        question="De transformer van de paper heeft 2 lagen, breedte 128 en 4 heads, en ongeveer 4 · 10⁵ parameters buiten zijn embeddings. Je ModernGPT met dezelfde 2 lagen, breedte 128 en 4 heads, met SwiGLU en RoPE: hoeveel?",
        options={
            "right": "Ongeveer evenveel, dicht bij 4 · 10⁵: SwiGLU heeft drie matrices, maar smallere",
            "more": "Ongeveer twee keer zoveel: SwiGLU heeft drie matrices waar de MLP van de paper er twee heeft",
            "fewer": "Veel minder: RoPE heeft geen parameters, en de tabel met plekken van de paper wel",
        },
        reveal=f"**Ongeveer evenveel: {C.number(C.NON_EMBEDDING_COURSE, 0, L.lang)}.** SwiGLU heeft drie matrices, maar elk is smaller: een hidden breedte van 8/3 keer 128 in plaats van 4 keer 128, dus heeft hij met ongeveer 8 · 128² getallen evenveel als de twee van de paper. Een tabel met plekken zit buiten de blokken, net als de embeddings, dus die telt hoe dan ook niet mee.",
    )
    guess_size
    return (guess_size,)


@app.cell(hide_code=True)
def text_reading():
    mo.md(r"""
    ## Leeswijzer

    De derde ronde van de paper, voor deze stap: Appendix A.1, ongeveer een bladzijde.

    1. **A.1.1, de data:** welke vergelijkingen, hoe elke als tokens geschreven wordt, en hoe de training data gekozen wordt.
    2. **A.1.2, het model en zijn optimalisatie:** elk getal van het model en van de optimizer. Schrijf elk op met waar het staat; je hebt ze allemaal nodig in oefening 3.
    3. **Wat er niet staat:** welke details van de transformer laat de paper open?

    De woorden van de paper naast die van de course en de code:

    | Paper | Course | Code |
    |---|---|---|
    | $\langle x \rangle \langle op \rangle \langle y \rangle \langle = \rangle \langle x \circ y \rangle$ | een vergelijking als vijf tokens | `[a, p, b, p + 1]`, en dan `c` |
    | training data fraction | het deel van alle vergelijkingen dat voor de training dient | `fraction` |
    | width | de grootte van de vector van elk token, $d$ | `width` |
    | optimization budget | het grootste aantal trainingsstappen | `steps` |
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **De data is een tabel.** Voor optellen modulo $p$ heeft de tabel een vergelijking $a + b = c$ voor elke $a$ en elke $b$ van 0 tot en met $p - 1$, met $c = (a + b) \bmod p$: voor $p = 97$ zijn dat $97^2 = 9.409$ vergelijkingen. Elk getal is een eigen token, zonder cijfers: het model ziet symbolen, en leert alleen van hoe ze samengaan.

    **De split is willekeurig, en vast.** Schud alle vergelijkingen met een seed, en neem de eerste `int(fraction * 9409)` voor de training; de rest is de validation data. Dezelfde seed geeft dezelfde split, dus een run kan herhaald worden.

    Hieronder staat de hele tabel voor $p = 97$, een cel per vergelijking: $a$ van boven naar beneden, $b$ van links naar rechts. Een gevulde cel is training data, een lege cel validation data. Hij opent bij 50%, met de cel $50 + 10$ aangewezen. Het schuifje zet de fraction; de knop neemt een andere seed.
    """)
    return


@app.cell(hide_code=True)
def play_table_cell():
    play_table = play.Table(
        C.training, lambda a, b: (a + b) % C.PAPER["p"], p=C.PAPER["p"], fraction=C.TABLE_FRACTION, point=list(C.POINT),
        decimal="," if L.lang == "nl" else ".",
        labels={"task_kick": L.w["task"], "task": "Wijs lege cellen aan: waar liggen de cellen met hetzelfde antwoord, en hoeveel daarvan zijn gevuld?",
                "alt": "De tabel van a + b modulo 97 als een raster van 97 bij 97 cellen, a van boven naar beneden en b van links naar rechts; de cellen van de training data zijn gevuld. De cel 50 + 10 is aangewezen: de cellen met zijn antwoord, 60, liggen op een diagonaal die aan de rand aan de andere kant doorloopt.",
                "train": "training data: het model ziet het antwoord", "val": "validation data: het model moet het vinden",
                "ring": "hetzelfde antwoord als de cel die je aanwijst; de andere cellen vervagen", "fraction_name": "fraction", "seed": "Andere seed", "of": "van de",
                "counts": "vergelijkingen in de training data", "point": "Wijs een cel aan, of gebruik de pijltjestoetsen.", "op": "+",
                "is_train": "training data", "is_val": "validation data: dit moet het model zelf vinden",
                "same": "hetzelfde antwoord in {n} cellen, waarvan {k} gevuld"})
    mo.ui.anywidget(play_table)
    return


@app.cell(hide_code=True)
def text_idea_rest():
    mo.md(r"""
    De cellen met hetzelfde antwoord liggen op één diagonaal: één meer bij $a$ en één minder bij $b$ houdt $a + b$ gelijk, en aan de rand loopt de diagonaal aan de andere kant door, zoals op een klok. Elk antwoord heeft 97 cellen; bij 50% zijn er daarvan gemiddeld ongeveer 48 gevuld. Het model ziet de lege cellen nooit. Optellen geeft het één kortere weg: $b + a$ heeft hetzelfde antwoord als $a + b$, dus een lege cel waarvan de spiegelcel, aan de andere kant van de lijn van linksboven naar rechtsonder, gevuld is, kan het goed krijgen uit die ene cel. Bij 50% heeft ongeveer de helft van de lege cellen er een; $50 + 10$ niet. Alle andere lege cellen kan het model alleen goed krijgen door de regel achter de diagonalen te vinden, niet door cellen te onthouden. Dat meet de validation accuracy: de spiegelcellen geven daar hooguit zo'n helft van, de regel de rest.

    **Het antwoord is het enige doel.** Het model leest $a$, de bewerking, $b$ en $=$, en zijn logits op de plek van $=$ zijn zijn gok voor $c$. De paper berekent de loss en de accuracy alleen daar.

    **Elke instelling uit de paper.** Elk getal van het model en de optimizer komt van de plek in de paper die het geeft, met die plek ernaast. Waar de paper zwijgt, is je keuze een **afwijking**: schrijf hem op, zodat een verschil in de uitkomsten later naar hem terug te voeren is.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.md(r"""
    ### Nieuwe woorden en tekens

    | Woord of teken | Wat het betekent | In code |
    |---|---|---|
    | **modulair rekenen** | rekenen op een klok van $p$ uur: na $p - 1$ komt weer 0 | `(a + b) % p` |
    | **training data fraction** | het deel van alle vergelijkingen van de tabel dat voor de training dient | `fraction` |
    | **afwijking** | een punt waarop je reproductie verschilt van de paper, gekozen of gedwongen, en opgeschreven | `# afwijking: ...` |
    | `random.shuffle(rows)` | zet de items van een lijst in een willekeurige volgorde, in de lijst zelf; na `random.seed(seed)` altijd dezelfde volgorde | `random.shuffle(rows)` |

    ### Recept

    Zo bouw je het experiment van een paper:

    1. Maak de data zoals de paper hem beschrijft: welke voorbeelden, welke tokens, welke split.
    2. Haal elke instelling van het model en de optimizer uit de paper, met waar hij staat.
    3. Schrijf op wat de paper openlaat, en wat je daar koos.
    4. Bouw het model met die instellingen, en vergelijk wat je kunt tellen met de paper, zoals zijn grootte.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Eén vergelijking van de tabel voor $p = 5$, als tokens, en de grootte van zijn split.
    """)
    return


@app.cell
def example_tokens():
    _p = 5
    _a, _b = 3, 4
    _c = (_a + _b) % _p
    print("tokens:", [_a, _p, _b, _p + 1], "antwoord:", _c)   # de bewerking is token p, het teken = is token p + 1
    print("vergelijkingen:", _p * _p, "voor de training bij 50%:", int(0.5 * _p * _p))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    $3 + 4 = 7$, en op een klok van 5 uur is dat 2. De getallen 0 tot en met 4 zijn de tokens 0 tot en met 4, dus de bewerking en het teken $=$ krijgen de volgende twee, 5 en 6: een vocabulary van $p + 2$ tokens. Van de 25 vergelijkingen is 50% gelijk aan 12,5, en `int` houdt 12.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_dataset():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De data

        Maak `equations(p)` af, elke `(a, b, c)` van de tabel met `c = (a + b) % p`, rij voor rij: `a` in de buitenste lus, `b` in de binnenste, en `dataset(p, fraction, seed)`, die ze na `random.seed(seed)` schudt met `random.shuffle` en de eerste `int(fraction * len(rows))` teruggeeft voor de training en de rest voor de validatie. De check roept `dataset(3, 0.5, 0)` aan, en andere. Als hij goed is, wordt hij in je project bewaard als `repro/data.py`.
        """),
        L.task("Maak `equations` en `dataset` af: de tabel van optellen modulo `p`, geschud en gesplitst."),
    ])
    return


@app.cell(hide_code=True)
def exercise_dataset():
    box_dataset = L.code_box("dataset", lines=17)
    box_dataset
    return (box_dataset,)


@app.cell(hide_code=True)
def check_dataset(box_dataset):
    result_dataset = L.check("dataset", box_dataset.value)
    result_dataset
    return (result_dataset,)


@app.cell(hide_code=True)
def hints_dataset():
    L.hints(
        "dataset",
        [
            "`equations` is twee loops, de ene in de andere, en één `append` per paar. Welke waarde krijgt `c`?",
            "`random.shuffle(rows)` verandert `rows` zelf en geeft niets terug. Dan een slice tot `n`, en een vanaf `n`.",
        ],
        why="Een split die van niets anders afhangt dan de seed, laat elke latere run, en iedereen anders, dezelfde training data gebruiken.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fraction():
    mo.md(r"""
    ### Oefening 2. Welke fraction?
    """)
    return


@app.cell(hide_code=True)
def choice_fraction_cell():
    choice_fraction = L.choice(
        "which-fraction",
        question="Sectie 3.1 zegt dat de vertraging typisch is \"for dataset sizes that were close to the minimal dataset size for which the network generalized\", en dat bij grotere \"the training and validation curves tend to track each other more closely\". Welke fraction neemt je reproductie voor optellen?",
        options={
            "most": "90%: hoe meer training data, hoe zekerder de uitkomst",
            "figure": "50%: de fraction van Figure 1",
            "right": f"{round(100 * C.FRACTION)}%: een klein deel van de tabel, waar de vertraging zou moeten blijken",
        },
        correct="right",
        why={
            "most": f"Met het grootste deel van de tabel om op te trainen, lopen de curves gelijk op: in de run van de course bij 90% haalde de training 99% op stap {C.FRACTIONS[0.9]['t']} en de validatie op stap {C.number(C.FRACTIONS[0.9]['v'], 0, L.lang)}, te vroeg voor een vertraging van 10 keer.",
            "figure": f"Figure 1 gaat over delen, en volgens de paper hebben ingewikkeldere bewerkingen meer data nodig. In de run van de course van optellen bij 50% haalde de training 99% op stap {C.FRACTIONS[0.5]['t']} en de validatie op stap {C.number(C.FRACTIONS[0.5]['v'], 0, L.lang)}: {round(C.FRACTIONS[0.5]['v'] / C.FRACTIONS[0.5]['t'])} keer zo laat, volgens je plan nog geen vertraging.",
            "right": f"De paper vindt de vertraging dicht bij de kleinste fraction die nog generaliseert. In de run van de course bij 30% haalde de training 99% op stap {C.FRACTIONS[0.3]['t']} en de validatie op stap {C.number(C.FRACTIONS[0.3]['v'], 0, L.lang)}: {round(C.FRACTIONS[0.3]['v'] / C.FRACTIONS[0.3]['t'])} keer zo laat.",
        },
    )
    choice_fraction
    return (choice_fraction,)


@app.cell(hide_code=True)
def text_exercise_settings():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De instellingen

        Vul `SETTINGS` in met de waarden van de paper, uit Appendix A.1.2, en de fraction van oefening 2. Schrijf elke zo dat Python hem leest: `1e-3` voor $10^{-3}$, een tuple voor de twee betas, en een geheel getal voor een aantal, zoals `100_000` of `10**5` voor $10^5$ (`1e5` is een float). Waar de paper een regel geeft, zoals voor de batch, reken die uit voor je data: de training data heeft duizenden vergelijkingen. Klopt het, dan wordt het in je project bewaard als `repro/settings.py`.
        """),
        L.task("Vul `SETTINGS` in: elke instelling van de paper, en je fraction."),
    ])
    return


@app.cell(hide_code=True)
def exercise_settings():
    box_settings = L.code_box("settings", lines=15)
    box_settings
    return (box_settings,)


@app.cell(hide_code=True)
def check_settings(box_settings):
    result_settings = L.check("settings", box_settings.value)
    result_settings
    return (result_settings,)


@app.cell(hide_code=True)
def hints_settings():
    L.hints(
        "settings",
        [
            "Appendix A.1.2 heeft twee zinnen met getallen: een over de transformer, een over de optimizer. Neem ze een voor een.",
            "De batch: 512, of de helft van de training data, wat kleiner is. Hoeveel trainingsvergelijkingen geeft je fraction?",
        ],
        why="Een instelling met de plek waar hij vandaan komt, kan iedereen controleren; een zonder is een gok die niemand terugvindt.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    Je `dataset` en je `SETTINGS`, uit je project, en je ModernGPT gemaakt met die instellingen, vóór elke training: hoe groot de data is, hoeveel parameters het model buiten zijn embeddings heeft, en één batch trainingsvergelijkingen erdoorheen. Waar een module van je ModernGPT ontbreekt of kapot is, springt de versie van de course in.
    """)
    return


@app.cell(hide_code=True)
def demo_build(result_dataset, result_settings):
    result_dataset, result_settings  # run again when an exercise saves a part of the kit
    _r = C.demo()
    if _r is None:
        _out = mo.md("*Maak eerst oefening 1 en 3 af: dan staan je data en je instellingen in je project, en gebruikt deze cel ze.*")
    elif isinstance(_r, str):
        _out = mo.md({"data": "*Je `repro/data.py` stopt met een fout. Heb je hem na de check veranderd? Verwijder hem, en check oefening 1 opnieuw.*",
                      "settings": "*Je `repro/settings.py` stopt met een fout. Heb je hem na de check veranderd? Verwijder hem, en check oefening 3 opnieuw.*",
                      "model": "*Je ModernGPT stopt met een fout met deze instellingen. Heb je `llm/modern_gpt.py` of `repro/settings.py` na zijn check veranderd? Verwijder het bestand dat je veranderde, en check zijn oefening opnieuw.*"}[_r])
    else:
        _s = _r["settings"]
        _stand = ("De versie van de course sprong in voor: " + ", ".join(f"`{_x}`" for _x in _r["stand_ins"]) + ".") if _r["stand_ins"] else ""
        _out = mo.md(f"""
| | van jou | wat je kunt verwachten |
|---|---|---|
| vergelijkingen voor de training | {C.number(_r['train'], 0, L.lang)} | {round(100 * _s['fraction'])}% van {C.number(_s['p'] ** 2, 0, L.lang)} |
| vergelijkingen voor de validatie | {C.number(_r['val'], 0, L.lang)} | de rest |
| parameters buiten de embeddings | {C.number(_r['params'], 0, L.lang)} | ongeveer 400.000 |
| een batch tokens | {_r['batch'][0]} × {_r['batch'][1]} | {_s['batch']} vergelijkingen van 4 tokens, zonder het antwoord |
| logits | {_r['logits'][0]} × {_r['logits'][1]} × {_r['logits'][2]} | een rij van {_s['p'] + 2} voor elke plek; de laatste gokt `c` |
| loss op de antwoorden, vóór de training | {C.number(_r['loss'], 2, L.lang)} | ln {_s['p'] + 2} = {C.number(_r['chance'], 2, L.lang)}, voor een model dat gelijk verdeeld gokt |

{_stand}
""")
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Met de instellingen van de paper kloppen de data en de grootte van het model met de paper. Drie delen van je model komen er niet uit: RMSNorm, RoPE en SwiGLU. De paper zegt alleen "a standard decoder-only transformer", dus elk is een afwijking: schrijf ze op in je kit, en houd ze in gedachten. Als je uitkomsten verschillen van die van de paper, zijn ze een van de eerste plekken om te kijken. De loss vóór de training ligt dicht bij die van een model dat gelijk verdeeld gokt over alle tokens: waar de volgende stap, de sanity checks, het eerst naar kijkt.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het plan uit [Begeleide reproductie 1](/course/open/repro-guided-claim?lang=nl), nu met zijn data en zijn instellingen; je ModernGPT uit [Project: je GPT een modern LLM maken](/course/open/llm-modern-gpt?lang=nl), aangepast aan de paper; en de zorg uit [Van formule naar code](/course/open/res-formula-to-code?lang=nl), voor elk getal van een paper. De volgende les, *Begeleide reproductie 3: sanity checks*, stuurt opnieuw één batch erdoor en laat het model die uit het hoofd leren, vóór elke lange run.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_dataset, choice_fraction, result_settings):
    L.wrap_up(result_dataset, choice_fraction, result_settings)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
