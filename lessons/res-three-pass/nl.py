# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les res-three-pass. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-three-pass", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [De anatomie van een paper over machine learning](/course/open/res-paper-anatomy?lang=nl) en [Het transformer block](/course/open/attn-transformer-block?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_claim():
    quick_claim = L.choice(
        "quick-claim",
        question="Je wilt in een minuut weten wat een paper beweert toe te voegen. Waar kijk je?",
        options={
            "method": "In de methode, waar het model beschreven wordt",
            "right": "In het abstract en de lijst met bijdragen",
            "appendix": "In de appendix, waar de details staan",
        },
        correct="right",
        why={
            "method": "De methode zegt in detail wat ze gedaan hebben; hij vat de claim niet samen. Zie [De anatomie van een paper over machine learning](/course/open/res-paper-anatomy?lang=nl).",
            "appendix": "De appendix bevat de details om het na te bouwen, niet de claim. Zie [De anatomie van een paper over machine learning](/course/open/res-paper-anatomy?lang=nl).",
            "right": "Het abstract zegt wat er beweerd wordt, en de bijdragen sommen op wat de rest van de paper moet onderbouwen.",
        },
    )
    quick_claim
    return (quick_claim,)


@app.cell(hide_code=True)
def quick_check_block():
    quick_block = L.choice(
        "quick-block",
        question="Wat gebeurt er in het transformer block dat je bouwde met `x`, vlak voordat het de attention in gaat?",
        options={
            "mlp": "Het gaat door het MLP",
            "add": "Het wordt opgeteld bij de uitvoer van de attention",
            "right": "Het gaat door een layer normalization",
        },
        correct="right",
        why={
            "mlp": "Het MLP komt na de attention, in de tweede tak. Zie [Het transformer block](/course/open/attn-transformer-block?lang=nl).",
            "add": "De optelling komt na de attention: `x + self.attention(self.norm1(x))`. Zie [Het transformer block](/course/open/attn-transformer-block?lang=nl).",
            "right": "Je block is pre-norm: `x + self.attention(self.norm1(x))`. Onthoud dat voor de derde ronde van deze les.",
        },
    )
    quick_block
    return (quick_block,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je hebt les voor les een GPT gebouwd. De paper waar zijn transformer blocks vandaan komen, *Attention Is All You Need* (2017), telt op arXiv zo'n vijftien pagina's, vol formules en tabellen. Lees je hem van voor naar achter, dan loop je vast bij de eerste formule die je niet volgt, of raak je halverwege paragraaf 3 de draad kwijt. Lezen in drie rondes, elk met een eigen doel en een eigen tijd, brengt je verder in minder tijd, en vertelt je wanneer je kunt stoppen.

    **Een voorbeeld uit het dagelijks leven.** Een huis kopen: eerst de foto's en de prijs online, vijf minuten, om te beslissen of een bezichtiging de moeite waard is. Dan een bezichtiging van een uur: past het, hoe ziet het er vanbinnen uit? En alleen voor het huis dat je echt wilt, een bouwkundige keuring die uren duurt, om te vinden wat er in de muren verborgen zit.
    """)
    return


@app.cell(hide_code=True)
def predict_time():
    guess_time = L.guess(
        "third-pass-time",
        question="S. Keshav, die de methode met drie rondes in 2007 opschreef, geeft elke ronde een eigen tijd. Hoe lang denk je dat de derde ronde, die door elk detail gaat, iemand kost die net begint met papers lezen?",
        options={
            "minutes": "Zo'n tien minuten",
            "hour": "Ongeveer een uur",
            "right": "Vier of vijf uur",
        },
        reveal="**Vier of vijf uur** voor een beginner, ongeveer een uur voor een ervaren lezer. De eerste ronde kost vijf tot tien minuten, de tweede hooguit een uur. Daarom doe je de derde ronde alleen voor de paar papers die je echt nodig hebt: veel papers leg je na de eerste weg.",
    )
    guess_time
    return (guess_time,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Bij **lezen in drie rondes** lees je een paper drie keer, elke keer dieper, en na elke ronde beslis je of je doorgaat.

        **De eerste ronde** is een snelle blik: de titel, het abstract, de introductie, de kopjes van de paragrafen en de conclusie. Daarna kun je vijf vragen beantwoorden, die Keshav de vijf C's noemt:

        1. **Category**, soort: wat voor paper is het? Gebruik de soorten bijdragen uit [Soorten papers en waar ze verschijnen](/course/open/res-how-research-is-published?lang=nl). <!-- termen:ok -->
        2. **Context**: op welk eerder werk bouwt hij voort?
        3. **Correctness**, juistheid: lijken zijn aannames te kloppen?
        4. **Contributions**, bijdragen: wat beweert hij toe te voegen?
        5. **Clarity**, helderheid: is hij goed geschreven?

        Dan beslis je: verder lezen, of wegleggen.

        **De tweede ronde** leest de hele paper zorgvuldig, maar slaat bewijzen en lange afleidingen over. Bekijk elke figuur en tabel zoals in [Een abstract, een figuur en een resultatentabel lezen](/course/open/res-figure-and-table?lang=nl), en streep de referenties aan die je niet gelezen hebt. Daarna kun je iemand anders vertellen wat de paper doet, en waarop dat rust.

        **De derde ronde** gaat door elk detail alsof je het werk zelf zou nabouwen: elke instelling, elke aanname, elke stap die de paper overslaat. Vergelijk elke keuze met wat jij gedaan zou hebben. Daarna zou je de hele paper uit je hoofd kunnen reconstrueren, en ken je zijn verborgen aannames en zijn zwakke plekken.
        """),
        mo.Html(C.passes({"title": "Drie rondes", "read": "Lees:", "after": "Daarna:",
                          "passes": [("Eerste ronde", "5 tot 10 minuten", "titel, abstract, introductie, kopjes, conclusie", "de vijf C's, en of je verder leest"),
                                     ("Tweede ronde", "hooguit een uur", "de hele tekst en elke figuur en tabel; nog geen bewijzen", "wat hij doet en waarop dat rust; de referenties om te lezen"),
                                     ("Derde ronde", "eerst 4 tot 5 uur, later ongeveer een uur", "elk detail, alsof je het zou nabouwen", "hoe je het nabouwt, en zijn verborgen aannames")],
                          "alt": "Drie kaartjes. Eerste ronde, 5 tot 10 minuten: lees de titel, het abstract, de introductie, de kopjes en de conclusie; daarna de vijf C's, en of je verder leest. Tweede ronde, hooguit een uur: lees de hele tekst en elke figuur en tabel, nog geen bewijzen; daarna wat de paper doet en waarop dat rust, en de referenties om te lezen. Derde ronde, eerst 4 tot 5 uur, later ongeveer een uur: elk detail, alsof je het zou nabouwen; daarna hoe je het nabouwt, en zijn verborgen aannames."})),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **lezen in drie rondes** | een paper drie keer lezen, elke keer dieper: een snelle blik, een zorgvuldige lezing, en elk detail |
        | **vijf C's** | de vragen na de eerste ronde: category, context, correctness, contributions, clarity <!-- termen:ok --> |
        """),
        L.watch_out("In de tweede ronde mag je overslaan wat je nog niet volgt: een bewijs, een afleiding, een formule. Schrijf op waar het staat, en kom erop terug in de derde ronde, als de paper die krijgt."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een paper te lezen:

    1. **Eerste ronde**, 5 tot 10 minuten: lees de titel, het abstract, de introductie, de kopjes en de conclusie. Beantwoord de vijf C's, en beslis of je verder leest.
    2. **Tweede ronde**, hooguit een uur: lees de hele tekst, zonder de bewijzen. Lees elke figuur en tabel zorgvuldig, en streep de referenties aan die je niet gelezen hebt. Schrijf in een paar zinnen op wat de paper doet, en waarop dat rust.
    3. **Derde ronde**, alleen voor een paper die je echt nodig hebt: ga door elk detail alsof je het zou nabouwen. Schrijf elke instelling en elke aanname op, en wat de paper weglaat.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De eerste ronde van [Attention Is All You Need](https://arxiv.org/abs/1706.03762), van Vaswani en zeven anderen, 2017. Open hem naast deze les als je wilt: het abstract, de introductie, de kopjes van paragraaf 1 tot en met 7, en de conclusie in paragraaf 7, kosten zo'n tien minuten.

    1. **Category:** een paper over een methode: een nieuw soort model, de transformer, vooral getest op vertalen. <!-- termen:ok -->
    2. **Context:** hij bouwt voort op modellen voor vertalen met een encoder en een decoder die een zin woord voor woord lezen, en op attention, die zulke modellen al gebruikten; de introductie noemt allebei.
    3. **Correctness:** de claim rust op scores voor vertalen op twee gangbare test sets uit 2014, Engels naar Duits en Engels naar Frans. De aanname: dat die scores meten hoe goed een vertaling is. Dat lijkt voorlopig redelijk; hoe de scores gemeten zijn, is voor een volgende ronde.
    4. **Contributions:** een model dat alleen uit attention bestaat, zonder woord voor woord te lezen; betere vertalingen dan de beste eerdere modellen; en veel minder trainingstijd.
    5. **Clarity:** kort en compact, met een helder plan in de kopjes: het model in paragraaf 3, het trainen in paragraaf 5, de resultaten in paragraaf 6.

    **Beslissing:** verder lezen. Voor jou is deze paper een tweede ronde waard: de meeste delen heb je zelf gebouwd.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pass():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Welke ronde?

    Zes dingen die je met een paper kunt doen. Zet elk bij de ronde waar het hoort.
    """)
    return


@app.cell(hide_code=True)
def exercise_pass():
    sort_pass = L.sort("which-pass")
    sort_pass
    return (sort_pass,)


@app.cell(hide_code=True)
def check_pass(sort_pass):
    result_pass = L.check("which-pass", sort_pass)
    result_pass
    return (result_pass,)


@app.cell(hide_code=True)
def hints_pass():
    L.hints(
        "which-pass",
        [
            "Vraag bij elke taak hoe diep hij gaat: een blik op het geheel in een paar minuten, een zorgvuldige lezing van de tekst en de figuren, of elk detail alsof je het zou nabouwen.",
            "Twee van de taken horen bij de snelle blik van de eerste ronde: de ene is wat je daarin leest, de andere noemt twee van de vijf C's. Twee vragen om de tekst en de tabellen zorgvuldig te lezen, maar niet om elke instelling.",
        ],
        why="**Eerste ronde:** de kopjes en de conclusie lezen en beslissen; de soort bijdrage en het eerdere werk noemen. **Tweede ronde:** de rijen van de resultatentabel controleren; de referenties aanstrepen die je nog nodig hebt. **Derde ronde:** uitzoeken waar elke layer normalization staat en vergelijken met je block; elke instelling opschrijven om het model zelf te trainen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_stuck():
    mo.md(r"""
    ### Oefening 2. Vast in de tweede ronde
    """)
    return


@app.cell(hide_code=True)
def choice_stuck_cell():
    choice_stuck = L.choice(
        "stuck",
        question="In je tweede ronde van de paper over de transformer kom je bij de formule voor de learning rate in paragraaf 5.3, en je volgt hem nog niet. Wat doe je?",
        options={
            "work": "Stoppen en de formule uitwerken tot je hem begrijpt, dan verder lezen",
            "right": "Opschrijven waar hij staat, verder lezen, en hem in de derde ronde uitwerken",
            "skip": "Hem voorgoed overslaan: de learning rate is een detail",
        },
        correct="right",
        why={
            "work": "Dat is de derde ronde. In de tweede mag een formule die je nog niet volgt je er niet van weerhouden de hele paper te zien. Zie het let-op in *Het idee*.",
            "skip": "Om het model na te bouwen doet de learning rate ertoe. Je stelt hem uit, je laat hem niet vallen. Zie *Het idee*.",
            "right": "De tweede ronde is voor het hele beeld. De formule, een warmup gevolgd door een langzame daling, is voor de derde ronde.",
        },
    )
    choice_stuck
    return (choice_stuck,)


@app.cell(hide_code=True)
def text_exercise_differences():
    mo.md(r"""
    ### Oefening 3. Een stukje van de derde ronde

    Open [de paper](https://arxiv.org/abs/1706.03762) en lees paragraaf 3.1, 3.4 en 5.4: een paar alinea's, samen minder dan een pagina. Vergelijk wat daar staat met de GPT die je bouwde.
    """)
    return


@app.cell(hide_code=True)
def explain_differences():
    explain_differences_cell = L.explain(
        "differences",
        question="Noem drie dingen die de transformer uit de paper doet en jouw GPT niet, of anders. Schrijf bij elk de paragraaf waar je het vond, en wat jouw GPT in plaats daarvan doet.",
        model_answer="**Paragraaf 3.1:** het model heeft twee stapels, een encoder en een decoder; jouw GPT heeft één stapel blocks met causale attention. **Paragraaf 3.1:** elke sub-layer geeft LayerNorm(x + Sublayer(x)): de layer normalization komt na de optelling; jouw block normaliseert vóór de attention en vóór het MLP. **Paragraaf 3.4:** de embeddings worden vermenigvuldigd met de wortel uit d_model, wat jouw GPT d noemt, en één matrix wordt gedeeld door de embeddings van de tokens in de encoder en de decoder en de layer vóór de softmax; jouw GPT heeft een eigen output head. **Paragraaf 5.4:** dropout van 0,1 op de uitvoer van elke sub-layer en op de embeddings, en label smoothing: het target voor het juiste token is ongeveer 0,9 in plaats van 1, en de rest wordt over de andere tokens verdeeld; jouw GPT heeft geen van beide.",
        points=[
            "je noemt drie verschillen, zoals de encoder, de plek van de layer normalization, de geschaalde en gedeelde embeddings, dropout of label smoothing",
            "bij elk noem je de paragraaf waar je het vond",
            "bij elk zeg je wat jouw GPT in plaats daarvan doet",
        ],
    )
    explain_differences_cell
    return (explain_differences_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [De anatomie van een paper over machine learning](/course/open/res-paper-anatomy?lang=nl) vertelde waar elk deel staat; de drie rondes vertellen hoe diep je elk deel leest, en wanneer je stopt. Je derde ronde vond verschillen tussen de paper en je GPT: elk is een keuze die iemand maakte, en sommige zijn later veranderd omdat een andere keuze beter werkte. Aan het eind van de course doe je de derde ronde echt, op de paper die je reproduceert. In de volgende les, *Eén ding tegelijk veranderen*, leer je hoe een experiment zo'n keuze toetst.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_stuck, explain_differences_cell, result_pass):
    L.wrap_up(result_pass, choice_stuck, explain_differences_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
