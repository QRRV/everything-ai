# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les res-find-papers. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-find-papers", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Soorten papers en waar ze verschijnen](/course/open/res-how-research-is-published?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_venue():
    quick_venue = L.choice(
        "quick-venue",
        question="Je vindt een paper twee keer: als versie 1 op arXiv, uit februari 2025, en later in de proceedings van ICML 2025. Wat is er in de tussentijd gebeurd?",
        options={
            "same": "Niets wat ertoe doet: het is dezelfde paper op een andere plek",
            "right": "Een paar reviewers beoordeelden hem, en de conferentie accepteerde hem",
            "repeated": "Andere onderzoekers herhaalden zijn resultaat",
        },
        correct="right",
        why={
            "same": "In de proceedings van een conferentie staan alleen de papers die ze na peer review accepteerde, dus de tweede plek zegt hoeveel controle de paper kreeg. Zie [Soorten papers en waar ze verschijnen](/course/open/res-how-research-is-published?lang=nl).",
            "repeated": "Een resultaat herhalen is een reproductie; in de proceedings van een conferentie staan betekent peer review. Zie [Soorten papers en waar ze verschijnen](/course/open/res-how-research-is-published?lang=nl).",
            "right": "ICML is een conferentie met peer review: een paar reviewers lazen en beoordeelden de paper. Dat is nog altijd geen garantie dat het resultaat klopt.",
        },
    )
    quick_venue
    return (quick_venue,)


@app.cell(hide_code=True)
def quick_check_kind():
    quick_kind = L.choice(
        "quick-kind",
        question="Een paper test 12 bestaande classifiers op nachtfoto's, meldt hoeveel slechter elk het doet dan bij daglicht, en stelt niets nieuws voor. Wat voor soort bijdrage is dat?",
        options={
            "method": "Een methode",
            "position": "Een position paper",
            "right": "Een empirische analyse",
        },
        correct="right",
        why={
            "method": "Een paper over een methode brengt een nieuwe manier om iets te doen, en deze paper stelt niets nieuws voor. Zie [Soorten papers en waar ze verschijnen](/course/open/res-how-research-is-published?lang=nl).",
            "position": "Een position paper verdedigt een standpunt; deze paper meet. Zie [Soorten papers en waar ze verschijnen](/course/open/res-how-research-is-published?lang=nl).",
            "right": "Hij meet hoe iets wat al bestaat zich gedraagt. Controleer dus hoeveel instellingen en runs, en of het patroon ook daarbuiten geldt.",
        },
    )
    quick_kind
    return (quick_kind,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je hebt een vraag, zoals: maakt het invullen van ontbrekende waarden met het gemiddelde een model slechter? Een gewone zoekmachine geeft je blogposts, advertenties en antwoorden van chatbots, waarvan sommige papers noemen die niet bestaan. Jij wilt de papers zelf: wie heeft dit gemeten, hoe, en wat kwam eruit. Met elke maand duizenden nieuwe papers heb je een manier nodig om de paar papers te vinden die ertoe doen.

    **Een voorbeeld uit het dagelijks leven.** Een goed boek heeft achterin een lijst met bronnen: daarmee vind je oudere boeken over hetzelfde onderwerp. Nieuwere boeken die het noemen, staan er niet in: het werd gedrukt voordat zij geschreven werden.
    """)
    return


@app.cell(hide_code=True)
def predict_citations():
    guess_citations = L.guess(
        "transformer-citations",
        question="Aan het eind noemt de paper over de transformer uit 2017 een paar dozijn referenties: het oudere werk waarop hij voortbouwt. Hoeveel latere papers denk je dat hem citeren, in 2026?",
        options={
            "dozens": "Ook een paar dozijn",
            "thousands": "Een paar duizend",
            "right": "Meer dan 100.000",
        },
        reveal="**Meer dan 100.000**, volgens de telling van Google Scholar in 2026. Zijn eigen referenties lees je in een minuut. De papers die hem citeren, zijn er meer dan iemand kan lezen, dus daar filter je: op jaar, op woorden in de titel, of op hoe vaak elk van die papers zelf weer geciteerd wordt.",
    )
    guess_citations
    return (guess_citations,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Literatuur zoeken** is de papers vinden die ertoe doen voor een vraag. Er zijn drie ingangen, en een goede zoektocht gebruikt ze alle drie.

        **Begin bij iemand die al gezocht heeft.** Een **survey** is een paper die de papers van een heel vakgebied ordent en vergelijkt; een leeslijst is een lijst papers die een docent of onderzoeker aanraadt voor een onderwerp. Zoek op je onderwerp met het woord "survey" erbij.

        **Zoek met zoektermen.** Een zoekmachine voor papers vindt ze via hun woorden, vooral die in de titel en het abstract, de korte samenvatting bovenaan. In 2026 zijn de bekendste Semantic Scholar en Google Scholar, voor bijna elk vakgebied, met bij elk resultaat hoe vaak het geciteerd is, en arXiv, voor de preprints die erop staan. Gebruik meer dan één term: onderzoekers noemen hetzelfde op verschillende manieren. Papers zijn bijna altijd in het Engels, dus zoek in het Engels: ontbrekende waarden heten daar *missing values* of *missing data*, en het invullen ervan *imputation*. <!-- termen:ok -->

        **Volg citaties.** Aan het eind noemt elke paper het werk dat hij citeert: zijn **referenties**. Die volgen gaat **achteruit**, naar ouder werk. De papers volgen die hem citeren, gaat **vooruit**, naar nieuwer werk. Een paper kan niet noemen wie hem later zal citeren, dus vooruit heb je een zoekmachine nodig die dat bijhoudt: Semantic Scholar en Google Scholar laten die papers zien onder *Citations* of *Cited by*, en op de pagina van elke paper op arXiv linkt het kader *References & Citations* naar sites die ze laten zien. Citaties ronde na ronde beide kanten op volgen, heet snowballing. <!-- termen:ok -->

        Alle papers samen, met een pijl van elke paper naar elke paper die hij citeert, vormen een **citatiegraaf**. Het plaatje laat een paar papers rond de transformer zien.
        """),
        mo.Html(C.around("transformer", {"title": "Rond de paper over de transformer", "forward": "Vooruit: papers die hem citeren (6 van meer dan 100.000)",
                                         "back": "Achteruit: zijn referenties (7 van een paar dozijn)", "cite": "↓ citeert",
                                         "alt": "De paper over de transformer uit 2017 in het midden. Erboven zes latere papers die hem citeren: gpt 2018, bert 2018, gpt2 2019, scaling 2020, gpt3 2020 en vit 2020. Eronder zeven van zijn referenties: align 2014, seq2seq 2014, lstm 1997, adam 2014, dropout 2014, resnet 2015 en layernorm 2016. De pijlen wijzen omlaag, van een paper naar de papers die hij citeert."})),
        mo.md(r"""
        Sommige vakgebieden schrijven een zoektocht zo precies op dat iedereen hem kan herhalen: de zoektermen, de zoekmachines, wanneer er gezocht is, en waarom elke paper bleef of afviel. Dat is een systematische review, en PRISMA is een veelgebruikte checklist om er verslag van te doen. Voor je eigen werk is een lichtere versie genoeg: schrijf je zoektermen op en waar je zocht.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **literatuur zoeken** | de papers vinden die ertoe doen voor een vraag: via surveys en leeslijsten, met zoektermen, en door citaties te volgen |
        | **survey** | een paper die de papers van een heel vakgebied ordent en vergelijkt; een goede plek om in een nieuw onderwerp te beginnen |
        | **referentie** | een paper die aan het eind van een andere paper genoemd wordt, als werk waarop die andere paper voortbouwt of dat hij bespreekt |
        | **citatie** | een paper die een andere noemt; hoe vaak een paper geciteerd is, is een ruwe maat voor zijn invloed |
        | **citatiegraaf** | alle papers, met een pijl van elke paper naar elke paper die hij citeert |
        | **achteruit, vooruit** | achteruit naar de referenties van een paper, ouder werk; vooruit naar de papers die hem citeren, nieuwer werk |
        """),
        L.watch_out("Chatbots kunnen referenties verzinnen: een titel die goed klinkt, de namen van echte onderzoekers, een jaartal, en geen paper die zo heet. Controleer elke paper die je niet zelf gevonden hebt op arXiv, Semantic Scholar of Google Scholar, voordat je hem gebruikt."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om de papers voor een vraag te vinden:

    1. Schrijf de vraag op in één zin, en kies twee tot vier zoektermen, met andere woorden voor hetzelfde.
    2. Zoek eerst naar een survey of een leeslijst: zoek op je termen met het woord "survey" erbij.
    3. Zoek op Semantic Scholar, Google Scholar of arXiv met je termen. Houd de twee of drie papers waarvan titel en abstract het best passen, en plaats elk met het recept van [Soorten papers en waar ze verschijnen](/course/open/res-how-research-is-published?lang=nl).
    4. Ga vanuit elke paper die je hield achteruit door zijn referenties en vooruit door de papers die hem citeren, en houd wat bij je vraag past.
    5. Doe stap 4 opnieuw met de papers die je hield, tot een ronde vooral papers oplevert die je al hebt. Schrijf je zoektermen op en waar je zocht.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 4 in code, op een klein stuk van de echte citatiegraaf. Voor elke paper bevat `_cites` een deel van de papers uit deze set die hij echt citeert; de tabel onder de cel zegt voor welke papers de namen staan. Niet elke pijl staat erin: we lieten er een paar weg, en elke paper citeert ook papers buiten deze set. Eén ding in de Python is nieuw: `in` vraagt ook of een waarde in een lijst staat, zoals het vraagt of een key in een dictionary staat. `"gpt" in ["transformer", "gpt"]` geeft `True`. Het commentaar zegt welke helft van stap 4 elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_round():
    # een klein stuk van de echte citatiegraaf: per paper een deel van de papers uit deze set die hij citeert
    _cites = {
        "lstm": [],
        "dropout": [],
        "adam": [],
        "seq2seq": ["lstm"],
        "align": ["seq2seq"],
        "resnet": [],
        "layernorm": [],
        "transformer": ["align", "seq2seq", "lstm", "adam", "dropout", "resnet", "layernorm"],
        "gpt": ["transformer", "adam", "layernorm"],
        "bert": ["transformer", "gpt"],
        "gpt2": ["transformer", "gpt", "bert"],
        "scaling": ["transformer", "gpt2"],
        "gpt3": ["transformer", "gpt2", "bert", "scaling"],
        "vit": ["transformer", "bert", "resnet"],
    }
    _paper = "gpt2"
    # 4. achteruit: de referenties van de paper
    _back = _cites[_paper]
    # 4. vooruit: elke paper die hem bij zijn referenties heeft
    _forward = []
    for _other in _cites:
        if _paper in _cites[_other]:
            _forward.append(_other)
    print("achteruit:", _back)
    print("vooruit:", _forward)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Achteruit vanuit `gpt2` (2019): de transformer, GPT en BERT, alle drie ouder. Vooruit: de scaling laws en GPT-3, allebei uit 2020, nieuwer. Achteruit is één keer opzoeken, want de lijst staat in de paper zelf. Vooruit is een loop over elke paper in de graaf nodig: in de echte graaf, met miljoenen papers, is dat het werk dat Semantic Scholar en Google Scholar voor je doen.

    | Naam | Paper, met het jaar waarin hij voor het eerst verscheen |
    |---|---|
    | `lstm` | Hochreiter en Schmidhuber (1997), *Long Short-Term Memory* |
    | `dropout` | Srivastava e.a. (2014), *Dropout* |
    | `adam` | Kingma en Ba (2014), *Adam* |
    | `seq2seq` | Sutskever e.a. (2014), *Sequence to Sequence Learning with Neural Networks* |
    | `align` | Bahdanau e.a. (2014), *Neural Machine Translation by Jointly Learning to Align and Translate* | <!-- termen:ok -->
    | `resnet` | He e.a. (2015), *Deep Residual Learning for Image Recognition* | <!-- termen:ok -->
    | `layernorm` | Ba e.a. (2016), *Layer Normalization* |
    | `transformer` | Vaswani e.a. (2017), *Attention Is All You Need* |
    | `gpt` | Radford e.a. (2018), *Improving Language Understanding by Generative Pre-Training* | <!-- termen:ok -->
    | `bert` | Devlin e.a. (2018), *BERT* |
    | `gpt2` | Radford e.a. (2019), *Language Models are Unsupervised Multitask Learners* |
    | `scaling` | Kaplan e.a. (2020), *Scaling Laws for Neural Language Models* |
    | `gpt3` | Brown e.a. (2020), *Language Models are Few-Shot Learners* |
    | `vit` | Dosovitskiy e.a. (2020), *An Image is Worth 16x16 Words* |
    """)
    return


@app.cell(hide_code=True)
def text_exercise_round():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Eén ronde, beide kanten op

        Maak `one_round(cites, paper)`: de papers die in de graaf `cites` één stap van `paper` liggen, achteruit en vooruit, samen in één lijst op alfabet. Het vak geeft je het begin en het eind; vervang de twee `...`. Aan het eind zet `back + forward` de twee lijsten achter elkaar: `["a"] + ["b"]` geeft `["a", "b"]`. De check roept `one_round(cites, "bert")` aan met de graaf van het uitgewerkte voorbeeld, en probeert ook andere papers en een andere graaf.
        """),
        L.task("Laat `one_round(cites, paper)` de referenties van `paper` en de papers die hem citeren teruggeven, op alfabet."),
    ])
    return


@app.cell(hide_code=True)
def exercise_round():
    box_round = L.code_box("one-round", lines=8)
    box_round
    return (box_round,)


@app.cell(hide_code=True)
def check_round(box_round):
    result_round = L.check("one-round", box_round.value)
    result_round
    return (result_round,)


@app.cell(hide_code=True)
def hints_round():
    L.hints(
        "one-round",
        [
            "Twee helften, zoals in het uitgewerkte voorbeeld, met `cites` en `paper` in plaats van `_cites` en `_paper`: achteruit is één keer opzoeken in `cites`; vooruit gaat langs elke paper in `cites`, met een eigen naam zoals `other`, en vraagt of `paper` in zijn lijst staat.",
            "`back = cites[___]`, dan `for other in cites:`, met daaronder `if paper in ___:` en daar weer onder `forward.append(___)`.",
        ],
        why="```python\ndef one_round(cites, paper):\n    back = cites[paper]\n    forward = []\n    for other in cites:\n        if paper in cites[other]:\n            forward.append(other)\n    return sorted(back + forward)\n```\n\n`cites[paper]` is de lijst met referenties, achteruit. De loop houdt elke paper waarvan de eigen lijst `paper` bevat: de papers die hem citeren, vooruit. `sorted` zet de twee lijsten samen op alfabet.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_direction():
    mo.md(r"""
    ### Oefening 2. Welke kant op?
    """)
    return


@app.cell(hide_code=True)
def choice_direction_cell():
    choice_direction = L.choice(
        "direction",
        question="Je vond een goede survey uit 2021 over het invullen van ontbrekende waarden. Nu wil je de papers over dat onderwerp van na 2021. Wat doe je?",
        options={
            "back": "De lijst met referenties achterin de survey een voor een doorlopen",
            "right": "Opzoeken wie de survey citeert, op Semantic Scholar of Google Scholar",
            "title": "Op arXiv zoeken naar de titel van de survey",
        },
        correct="right",
        why={
            "back": "De referenties zijn ouder dan de survey: achteruit leidt naar werk van vóór 2021. Zie *Het idee*.",
            "title": "Zo vind je de survey zelf terug, niet het werk dat erna kwam. Zie *Het idee*.",
            "right": "Papers die de survey citeren, kwamen erna: vooruit. Filter ze op jaar, of sorteer ze op hoe vaak ze geciteerd zijn.",
        },
    )
    choice_direction
    return (choice_direction,)


@app.cell(hide_code=True)
def text_exercise_plan():
    mo.md(r"""
    ### Oefening 3. Plan een zoektocht

    Je vraag: **maakt het invullen van ontbrekende waarden met het gemiddelde van hun kolom een model slechter dan die rijen weglaten?**
    """)
    return


@app.cell(hide_code=True)
def explain_plan():
    explain_plan_search = L.explain(
        "plan-search",
        question="Schrijf een zoekplan met stap 1 tot en met 4 van het recept: je zoektermen, met andere woorden voor hetzelfde; waar je begint; waar je zoekt; en hoe je citaties volgt.",
        model_answer="**Termen:** *missing values*, *missing data*, *imputation*, *mean imputation*; voor rijen weglaten *listwise deletion* of *complete-case analysis*; en het soort data, zoals *tabular data*. **Begin:** zoek op Semantic Scholar of Google Scholar naar *missing data imputation survey*, en gebruik het overzicht van de survey. **Zoeken:** dezelfde termen zonder *survey*; houd de twee of drie papers waarvan titel en abstract het best passen. **Citaties volgen:** achteruit door hun referenties voor de oudere, bekende methodes, en vooruit door de papers die hen citeren voor nieuwere vergelijkingen; stop als een ronde vooral papers oplevert die je al hebt.",
        points=[
            "je noemt meer dan één zoekterm, met een ander woord voor hetzelfde, zoals *missing data* of *imputation*",
            "je begint bij een survey of een leeslijst",
            "je volgt citaties beide kanten op: achteruit door de referenties, vooruit door de papers die hen citeren",
        ],
    )
    explain_plan_search
    return (explain_plan_search,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Soorten papers en waar ze verschijnen](/course/open/res-how-research-is-published?lang=nl) leerde je een paper te plaatsen; nu kun je ze vinden. Citaties volgen komt terug als je aan het eind van de course een paper kiest om te reproduceren: de papers die hem citeren, laten zien of anderen er al op voortbouwden, of hem al probeerden te herhalen. In de volgende les, *De anatomie van een paper over machine learning*, kijk je in een paper: welke vraag elk deel beantwoordt, en waar de details staan.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_direction, explain_plan_search, result_round):
    L.wrap_up(result_round, choice_direction, explain_plan_search)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
