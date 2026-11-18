# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-reason-codes. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-reason-codes", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Soorten uitleg](/course/open/rai-kinds-of-explanation?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_kind():
    quick_kind = L.choice(
        "quick-kind",
        question="Voor één aanvrager vermenigvuldig je elke weight van een logistische regressie met de waarde van die aanvrager. Wat voor uitleg is dat?",
        options={
            "global": "Een global explanation, uit het model",
            "right": "Een local explanation, uit het model",
            "after": "Een local explanation, achteraf gemaakt",
        },
        correct="right",
        why={
            "global": "Alleen de weights zijn global: voor iedereen hetzelfde. Weight keer waarde gaat over één aanvrager: een local explanation, direct uit het model gelezen. Zie [Soorten uitleg](/course/open/rai-kinds-of-explanation?lang=nl).",
            "right": "Het gaat over één aanvrager, dus local, en het gebruikt de eigen weights van het model, dus het komt uit het model zelf: er wordt niets achteraf gemaakt.",
            "after": "Het is local, maar er wordt niets achteraf gemaakt: je leest de eigen weights van het model. Een post-hoc explanation is een aparte methode die een getraind model bestudeert. Zie [Soorten uitleg](/course/open/rai-kinds-of-explanation?lang=nl).",
        },
    )
    quick_kind
    return (quick_kind,)


@app.cell(hide_code=True)
def quick_check_add():
    quick_add = L.choice(
        "quick-add",
        question="In het kredietmodel van de vorige les heeft `missed_payments` de weight −1,2. Aanvrager A miste 2 betalingen. Wat draagt `missed_payments` bij aan de logit van A?",
        options={
            "right": "−2,4",
            "weight": "−1,2, de weight",
            "value": "2, de waarde van A",
        },
        correct="right",
        why={
            "right": "Weight keer waarde: −1,2 · 2 = −2,4. Van de drie features van A woog deze het zwaarst tegen een ja.",
            "weight": "De weight is voor elke aanvrager hetzelfde. Wat hij voor A bijdraagt, is de weight keer de waarde van A: −1,2 · 2 = −2,4. Zie [Soorten uitleg](/course/open/rai-kinds-of-explanation?lang=nl).",
            "value": "De waarde alleen zegt niets over het model. Wat een feature bijdraagt, is de weight keer de waarde: −1,2 · 2 = −2,4. Zie [Soorten uitleg](/course/open/rai-kinds-of-explanation?lang=nl).",
        },
    )
    quick_add
    return (quick_add,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Stel dat een bank je lening weigert. In de Verenigde Staten moet de bank je dan vertellen waarom. Regulation B, de regels onder de Equal Credit Opportunity Act, vraagt om een verklaring met specifieke redenen, die de belangrijkste reden of redenen noemt. Zeggen dat je de vereiste score in het scoresysteem van de kredietverstrekker niet haalde, is volgens de regels onvoldoende. In de Europese Unie geeft de AVG iemand over wie uitsluitend langs geautomatiseerde weg een besluit is genomen met rechtsgevolgen of een vergelijkbaar aanmerkelijk effect, recht op "nuttige informatie over de onderliggende logica", en in 2025 noemde het Hof van Justitie van de EU dat een echt recht op uitleg. Een model dat over een lening beslist, moet dus een paar redenen meegeven die een mens kan volgen. Dat is een local explanation, omgezet in woorden.

    **Een voorbeeld uit het dagelijks leven.** Je krijgt een toets terug met een onvoldoende. Dat zegt wat, niet waarom. "Je verloor de meeste punten op breuken, daarna op procenten" zegt wat je moet oefenen, en het meet je af aan een voldoende: waarom deze uitslag, en niet de andere.
    """)
    return


@app.cell(hide_code=True)
def predict_count():
    guess_count = L.guess(
        "count",
        question="Hoeveel redenen moet een kredietverstrekker volgens Regulation B meegeven met een nee?",
        options={
            "four": "Precies vier",
            "most": "Hoogstens vier",
            "none": "Geen vast aantal",
        },
        reveal="**Geen vast aantal.** De officiële toelichting zegt dat de regels geen vast aantal redenen voorschrijven, maar dat meer dan vier redenen de aanvrager waarschijnlijk niet helpt. Een harde grens van vier komt uit een andere wet, de Fair Credit Reporting Act: bij een kredietscore krijg je hoogstens vier *key factors*, vijf als het aantal opvragingen van je gegevens er een van is. En die key factors tellen niet als de specifieke redenen waar Regulation B om vraagt.",
    )
    guess_count
    return (guess_count,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Redenen uit een model.** In de praktijk stuurt een kredietverstrekker zijn redenen als korte, vaste zinnetjes, elk met een code: reason codes. Die naam komt uit de praktijk van de sector; de wet spreekt van specifieke redenen en de belangrijkste reden of redenen. De redenen moeten de echte zijn: ze moeten de factoren beschrijven die de kredietverstrekker werkelijk meewoog of scoorde, ook een factor waarvan de aanvrager het verband met terugbetalen misschien niet ziet, zoals de leeftijd van de auto. Een reden mag alleen de factor noemen; wat eraan mis was, hoeft er niet bij: de brief mag "woonduur" zeggen in plaats van "te korte woonduur".

        **Afgemeten aan wie?** Regulation B schrijft niet voor hoe een kredietverstrekker met een scoremodel zijn redenen kiest, maar de officiële toelichting geeft twee methoden die volstaan. Beide zoeken de factoren waarop de score van de aanvrager het verst onder het gemiddelde bleef. Methode 1 neemt het gemiddelde van de aanvragers met een totaalscore op of net boven de laagste score voor een ja, methode 2 het gemiddelde van alle aanvragers. Deze course leest de score van een factor in een logistische regressie als wat hij bijdraagt aan de logit: zijn weight keer zijn waarde, zoals in [Soorten uitleg](/course/open/rai-kinds-of-explanation?lang=nl). Het verschil van een feature is dan zijn weight keer de waarde van de aanvrager min het gemiddelde van de referentiegroep, dus `weight * (value - average)`. Een verschil onder 0 telt tegen de aanvrager, en het meest negatieve verschil is de belangrijkste reden.

        **Waarom dit, en niet dat.** Tim Miller (2019) zette op een rij wat de filosofie en de psychologie weten over hoe mensen iets uitleggen. Een uitleg is contrastief: mensen vragen niet waarom P gebeurde, maar waarom P gebeurde in plaats van Q. Mensen kiezen: uit veel oorzaken pikken ze er één of twee. En oorzaken werken beter dan getallen: verwijzen naar kansen of statistische verbanden werkt minder goed dan verwijzen naar oorzaken. Het boek van Molnar maakt daar een advies van: geef maar 1 tot 3 redenen, ook als de wereld ingewikkelder is. Redenen die zijn afgemeten aan een referentiegroep zijn in die zin contrastief: waarom een nee voor jou, en niet de ja van een aanvrager die het net haalde. Die lezing is van deze course; de wet gebruikt het woord niet.

        **Een recht op uitleg?** De AVG geeft iemand het recht niet te worden onderworpen aan een besluit dat uitsluitend op geautomatiseerde verwerking berust en rechtsgevolgen heeft of iemand op een vergelijkbare manier aanmerkelijk treft, met uitzonderingen (artikel 22). Wordt zo'n besluit genomen, dan krijgt de persoon "nuttige informatie over de onderliggende logica" (artikelen 13 tot en met 15). Een recht op uitleg staat niet met zoveel woorden in die artikelen. Overweging 71 noemt het wel, een van de genummerde overwegingen aan het begin van de wet, met de automatische afwijzing van een online kredietaanvraag als voorbeeld. In 2017 betoogden Sandra Wachter, Brent Mittelstadt en Luciano Floridi dat de AVG alleen een recht geeft om geïnformeerd te worden. Andrew Selbst en Julia Powles antwoordden: dit is een recht op uitleg, of je het zo noemt of niet.

        **Wat de rechters zeiden.** In februari 2025 deed het Hof van Justitie van de EU uitspraak in een zaak over een kredietbeoordeling door het bedrijf Dun & Bradstreet Austria (zaak C-203/22): artikel 15, lid 1, onder h, geeft een echt recht op uitleg. Een ingewikkelde wiskundige formule overhandigen is niet genoeg. Wat het Hof wel transparant genoeg vond: de persoon vertellen in hoeverre andere persoonsgegevens tot een ander resultaat hadden geleid. Dat is een contrastive explanation. Een bedrijf dat zijn model een bedrijfsgeheim noemt, moet de informatie geven aan de toezichthouder of de rechter, die de belangen afweegt en beslist hoeveel de persoon krijgt. In 2023 droeg het Gerechtshof Amsterdam het bedrijf Ola op zijn chauffeurs ten minste toe te lichten "op basis van welke factoren en welke weging van die factoren" het tot hun fraud probability score, hun earning profile en de toedeling van ritten kwam, op straffe van een dwangsom van 3.000 euro per dag; een uiteenzetting van het volledige algoritme hoefde niet. En artikel 86 van de AI Act geeft iemand die aanmerkelijk geraakt wordt door een besluit op basis van een AI-systeem met een hoog risico uit bijlage III, waaronder kredietscores, recht op een duidelijke en zinvolle uitleg over de rol van het AI-systeem, maar alleen als ander EU-recht dat recht nog niet geeft.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **reason code** | een van de korte, vaste redenen die een kredietverstrekker meestuurt met een nee, elk met de naam van een factor die tegen de aanvrager woog; de wet zegt specifieke redenen |
        | **contrastive explanation** | een uitleg waarom de uitkomst dit was en niet iets anders, zoals waarom een nee en geen ja |
        | **recht op uitleg** | het recht van iemand over wie een geautomatiseerd besluit is genomen om te horen hoe het tot stand kwam; in de EU rust het op de AVG, en voor besluiten op basis van de AI-systemen met een hoog risico uit bijlage III ook op de AI Act |
        """),
        L.watch_out("Verschillende methoden kunnen verschillende redenen geven. Voor de aanvrager van deze les noemen de twee methoden van Regulation B een andere belangrijkste reden: de leeftijd tegen alle aanvragers, de opleiding tegen de aanvragers net boven de grens. En een reden is maar zo waar als de methode erachter. In 2022 schreef het Consumer Financial Protection Bureau (CFPB), de Amerikaanse toezichthouder voor financiële producten voor consumenten, dat uitleg achteraf een model maar benadert, en dat kredietverstrekkers moeten kunnen nagaan of die benadering klopt. Het CFPB trok die circulaire op 12 mei 2025 in; de regel die ze uitlegde, specifieke redenen die de echte factoren beschrijven, veranderde niet."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo vind je de redenen voor een nee van een logistische regressie:

    1. Kies een referentiegroep: de aanvragers op of net boven de grens, of alle aanvragers.
    2. Neem van elke feature het gemiddelde over die groep.
    3. Reken voor elke feature het verschil uit: zijn weight keer (de waarde van de aanvrager min dat gemiddelde).
    4. De features met een verschil onder 0 zijn de redenen, het meest negatieve eerst. Geef de paar die het zwaarst wegen; de factor noemen is genoeg.
    """)
    return


@app.cell(hide_code=True)
def picture_references():
    mo.vstack([
        mo.md("**Twee referentiegroepen, twee belangrijkste redenen**"),
        mo.Html(C.two_references("nl")),
        mo.md(r"""
        Eén aanvrager: 30 jaar, een middelbareschooldiploma, 40 uur werk per week. Links de verschillen tegen het gemiddelde van alle 48.842 aanvragers: de leeftijd weegt het zwaarst tegen. Rechts tegen het gemiddelde van de 1.608 aanvragers net boven de grens, die ouder zijn, hoger opgeleid en meer uren werken: nu weegt de opleiding het zwaarst tegen. Regulation B staat beide methoden toe.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het model van deze les is van de course zelf. Het gebruikt de tabel UCI Adult, die ook in [De 80%-regel](/course/open/rai-80-percent-rule?lang=nl) voorkomt: 48.842 mensen uit een Amerikaans onderzoek van 1994. Uit drie kolommen, `age`, `education_num` en `hours_per_week`, voorspelt het wie meer dan 50.000 dollar per jaar verdient, en we lezen zijn nee als het nee van een kredietverstrekker. `education_num` is het opleidingsniveau als getal van 1 tot 16: 9 is een middelbareschooldiploma in de VS, 13 een bachelor. `.to_numpy()` maakt van de drie kolommen een NumPy-array, met een rij per persoon. `LogisticRegression` uit scikit-learn werkt als `LinearRegression` in [Hetzelfde model met scikit-learn](/course/open/ml-sklearn?lang=nl); zijn `coef_` is een tabel met één rij, dus de weights zijn `coef_[0]`. Voer de cel uit: hij traint het model, rondt de weights af, en rekent de verschillen van één aanvrager uit tegen alle aanvragers, methode 2. Dat duurt een seconde of twee.
    """)
    return


@app.cell
def example_gaps():
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    # 48.842 mensen uit een Amerikaans onderzoek van 1994 (UCI Adult)
    _people = pd.read_csv("data/adult.csv.gz")
    _names = ["age", "education_num", "hours_per_week"]
    _X = _people[_names].to_numpy()
    _y = _people["income"] == ">50K"

    # het model: een logistische regressie op drie kolommen, de weights afgerond
    _model = LogisticRegression(max_iter=1000)   # meer stappen dan standaard, zodat het klaar is met leren
    _model.fit(_X, _y)
    _weights = _model.coef_[0].round(3)
    _bias = round(_model.intercept_[0], 2)
    print("weights", _weights)
    print("bias", _bias)

    # een aanvrager van 30, met een middelbareschooldiploma, die 40 uur per week werkt
    _person = np.array([30, 9, 40])
    print("logit", round(_person @ _weights + _bias, 3))

    # methode 2: de verschillen tegen het gemiddelde van alle aanvragers
    _gaps = _weights * (_person - _X.mean(axis=0))
    print("verschillen", _gaps.round(3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De weights zijn 0,046 per levensjaar, 0,345 per opleidingsniveau en 0,042 per uur, met de bias −8,48. De logit van de aanvrager is −2,315: een nee. Het model zegt ja tegen 5.831 van de 48.842 mensen; zijn ja of nee klopt bij 78,9 procent van alle 48.842; altijd nee zeggen zou bij 76,1 procent goed zitten. Een zwak model, maar genoeg om te laten zien hoe je redenen kiest.

    Tegen alle aanvragers weegt de leeftijd het zwaarst tegen deze aanvrager (−0,398), dan de opleiding (−0,372); de uren tellen nauwelijks (−0,018). Dat is methode 2. Oefening 1 doet methode 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_gaps():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Tegen de aanvragers net boven de grens

        Het vak begint met de tabel, het model en de aanvrager uit het uitgewerkte voorbeeld. `logits` bevat de logit van elke aanvrager, en `near` is `True` voor de 1.608 aanvragers met een logit van 0 tot 0,25: zij kregen een ja, maar net. De `&` in die regel combineert twee arrays van `True` en `False` element voor element: alleen `True` waar beide `True` zijn. Reken `gaps` uit met methode 1. Eerst `reference`: het gemiddelde van elke kolom over de aanvragers in `near`. Dan `gaps`: elke weight keer (de waarde van de aanvrager min dat gemiddelde). De check probeert ook een aanvrager van 52.
        """),
        L.task("Reken `reference` uit, het gemiddelde van elke kolom over de aanvragers in `near`, en dan `gaps`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gaps():
    box_gaps = L.code_box("gaps", lines=21)
    box_gaps
    return (box_gaps,)


@app.cell(hide_code=True)
def check_gaps(box_gaps):
    result_gaps = L.check("gaps", box_gaps.value)
    result_gaps
    return (result_gaps,)


@app.cell(hide_code=True)
def hints_gaps():
    L.hints(
        "gaps",
        [
            "Twee stappen: eerst de gemiddelden van de referentiegroep, dan de verschillen, zoals in de laatste regels van het uitgewerkte voorbeeld.",
            "`X[near]` houdt de rijen van de aanvragers in `near`, en `.mean(axis=0)` neemt het gemiddelde van elke kolom. Vermenigvuldig dan `weights` met `person - reference`.",
            "`reference = X[near].mean(___)`",
        ],
        why="`reference = X[near].mean(axis=0)`, dan `gaps = weights * (person - reference)`. Dat geeft leeftijd `-0.789`, opleiding `-1.344` en uren `-0.302`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_reasons():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Van verschillen naar redenen

        `gaps` bevat je uitkomst van oefening 1, afgerond, en `names` de drie features. Maak de lijst `reasons`: de namen van de features met een verschil onder 0, het meest negatieve eerst. `np.argsort(gaps)` geeft de posities van de verschillen van klein naar groot, zoals in [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl). De check probeert ook de verschillen van de aanvrager van 52.
        """),
        L.task("Maak `reasons`: de namen van de features met een verschil onder 0, het meest negatieve eerst."),
    ])
    return


@app.cell(hide_code=True)
def exercise_reasons():
    box_reasons = L.code_box("reasons", lines=11)
    box_reasons
    return (box_reasons,)


@app.cell(hide_code=True)
def check_reasons(box_reasons):
    result_reasons = L.check("reasons", box_reasons.value)
    result_reasons
    return (result_reasons,)


@app.cell(hide_code=True)
def hints_reasons():
    L.hints(
        "reasons",
        [
            "Loop over de verschillen van het meest negatieve naar boven, en houd de naam van elk verschil onder 0.",
            "Begin met `reasons = []`. Loop met `for i in np.argsort(gaps):`, en voeg daarbinnen `names[i]` toe als `gaps[i] < 0`.",
            "`reasons.append(___)`",
        ],
        why="`reasons = []`, dan `for i in np.argsort(gaps):` met `if gaps[i] < 0:` en daarbinnen `reasons.append(names[i])`. Dat geeft `['education_num', 'age', 'hours_per_week']`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_contrastive():
    mo.md(r"""
    ### Oefening 3. Waar mensen om vragen
    """)
    return


@app.cell(hide_code=True)
def choice_contrastive_cell():
    choice_contrastive = L.choice(
        "contrastive",
        question="Een aanvrager vraagt de bank: waarom kreeg ik een nee? Welk antwoord past het best bij wat Miller vond over wat mensen van een uitleg willen?",
        options={
            "all": "Alle drie de weights en de bias, samen met je logit",
            "right": "De twee features waarop je achterbleef bij een ja",
            "probability": "Je kans op een ja was 0,09",
        },
        correct="right",
        why={
            "all": "Volledig, maar niet selectief: Miller vond dat mensen één of twee oorzaken kiezen, en Molnar raadt 1 tot 3 redenen aan. En het zegt niet waarom een nee en geen ja. Zie *Waarom dit, en niet dat* in *Het idee*.",
            "right": "Kort en contrastief: een paar redenen, afgemeten aan de aanvragers die een ja kregen. Daar vragen mensen om volgens Miller, en dat geven jouw `reasons` uit oefening 2.",
            "probability": "Een kans zegt hoe zeker het model is, niet waarom. Volgens Miller werkt verwijzen naar kansen of statistische verbanden minder goed dan verwijzen naar oorzaken. Zie *Waarom dit, en niet dat* in *Het idee*.",
        },
    )
    choice_contrastive
    return (choice_contrastive,)


@app.cell(hide_code=True)
def text_exercise_court():
    mo.md(r"""
    ### Oefening 4. Waar het Hof om vroeg
    """)
    return


@app.cell(hide_code=True)
def choice_court_cell():
    choice_court = L.choice(
        "court",
        question="In 2025 deed het Hof van Justitie van de EU uitspraak over een kredietbeoordeling langs geautomatiseerde weg. Wat krijgt de beoordeelde persoon volgens artikel 15, lid 1, onder h, van de AVG?",
        options={
            "formula": "De volledige formule van het model",
            "right": "Hoe de gegevens tot de uitkomst leidden",
            "secret": "Niets, als het model een bedrijfsgeheim is",
        },
        correct="right",
        why={
            "formula": "Het Hof zei dat een ingewikkelde wiskundige formule overhandigen niet genoeg is: dat is geen beknopte en begrijpelijke uitleg. Zie *Wat de rechters zeiden* in *Het idee*.",
            "right": "De persoon krijgt uitleg over de procedure en de principes die echt op de eigen gegevens zijn toegepast, bijvoorbeeld in hoeverre andere persoonsgegevens tot een ander resultaat hadden geleid: een contrastive explanation.",
            "secret": "Een bedrijfsgeheim maakt geen einde aan het recht: het bedrijf moet de informatie geven aan de toezichthouder of de rechter, die de belangen afweegt en beslist hoeveel de persoon krijgt. Zie *Wat de rechters zeiden* in *Het idee*.",
        },
    )
    choice_court
    return (choice_court,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Soorten uitleg](/course/open/rai-kinds-of-explanation?lang=nl) splitste je een logit in wat elke feature bijdraagt: een local explanation. In deze les mat je die delen af aan een referentiegroep en maakte je van de meest negatieve redenen, kort en contrastief, zoals een kredietverstrekker ze meestuurt met een nee. De volgende les, [Glass-box models](/course/open/rai-glass-box-models?lang=nl), gaat over modellen waarvan een mens elke stap kan volgen. Later in deze module stelt [Wat zou er moeten veranderen?](/course/open/rai-counterfactuals?lang=nl) de vraag van de andere kant: welke verandering had een ja gegeven?
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_contrastive, choice_court, result_gaps, result_reasons):
    L.wrap_up(result_gaps, result_reasons, choice_contrastive, choice_court)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
