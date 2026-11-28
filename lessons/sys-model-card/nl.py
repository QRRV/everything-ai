# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-model-card. Route: course/route/lessen/systemen.yaml. Checks, de lezer, de cijfers, de omstandigheden en de plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-model-card", "nl")
    C = L.checks   # de cijferlezer, de 2.000 testcijfers, de omstandigheden, de checks en de plaatjes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl) en [Wie staat er in de data, en wie ontbreekt?](/course/open/rai-bias-in-data?lang=nl)
    """)
    return


@app.cell(hide_code=True)
def quick_check_card():
    quick_card = L.choice(
        "quick-card",
        question="Een model op een model hub leest 97 van de 100 testcijfers goed, staat op de card. Waar kijk je op de card naar, voor je het model in een app gebruikt?",
        options={
            "score": "Nergens meer naar: met 97 van de 100 goed is het model goed genoeg voor bijna elke app",
            "right": "Van welke data het model leerde en waarop het getest is, en of de licentie jouw gebruik toestaat",
            "size": "Hoeveel parameters het model heeft, want een model met meer parameters leest altijd beter",
            "date": "Wanneer het model getraind is, want een model van een tijd terug weet minder dan een nieuwer model",
        },
        correct="right",
        why={
            "right": "Een score geldt voor de testcijfers waarop de score gemeten is. In Een pretrained model gebruiken paste een card met witte cijfers op zwart niet bij foto's van cijfers op papier, en een licentie met NC niet bij een app die je verkoopt.",
            "score": "97 van de 100 geldt voor de eigen testcijfers van de card: lijken jouw beelden daarop, en mag je het model in je app gebruiken? Zie [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl).",
            "size": "Meer parameters zegt niets over of een model bij jouw beelden of jouw gebruik past: wat op de card zegt dat wel? Zie [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl).",
            "date": "Een model weet niet meer omdat het nieuwer is: wat op de card zegt of het model bij jouw beelden en jouw gebruik past? Zie [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl).",
        },
    )
    quick_card
    return (quick_card,)


@app.cell(hide_code=True)
def quick_check_mean():
    quick_mean = L.choice(
        "quick-mean",
        question="Een test set heeft 900 cijfers van groep A en 100 van groep B. Een lezer leest 95% van groep A goed en 75% van groep B. Wat is de accuracy over de hele test set?",
        options={
            "half": "85%: het gemiddelde van 95% en 75%, omdat elk van de twee groepen één keer telt",
            "a": "95%: groep A is negen tiende van de test set, dus telt alleen groep A",
            "right": "93%: in het gemiddelde telt groep A negen keer zo zwaar als groep B",
            "b": "75%: een accuracy is zo goed als de accuracy van de zwakste groep",
        },
        correct="right",
        why={
            "right": "0,9 × 95% + 0,1 × 75% = 93%. De totale score ligt dicht bij de score van groep A, en verbergt dat een op de vier cijfers van groep B fout gaat.",
            "half": "Elk cijfer telt één keer, niet elke groep: hoe zwaar telt een groep van 100 tegen een groep van 900? Zie [Wie staat er in de data, en wie ontbreekt?](/course/open/rai-bias-in-data?lang=nl)",
            "a": "Groep B is een tiende van de test set, en telt voor een tiende mee: wat is 0,9 × 95% + 0,1 × 75%? Zie [Wie staat er in de data, en wie ontbreekt?](/course/open/rai-bias-in-data?lang=nl)",
            "b": "De totale accuracy is een gemiddelde over alle cijfers, waarin elke groep meetelt naar het aantal cijfers: wat geeft dat hier? Zie [Wie staat er in de data, en wie ontbreekt?](/course/open/rai-bias-in-data?lang=nl)",
        },
    )
    quick_mean
    return (quick_mean,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De card van de cijferlezer uit [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl) zegt: 929 van de eerste 1.000 testbeelden van QMNIST goed. Stel dat je de cijfers wilt lezen die volwassenen op formulieren schrijven. Dan lees je deze regel en verwacht je ongeveer 7 fouten per 100 cijfers. Maar wie schreef deze 1.000 cijfers? Dat staat niet op de card. En een model doet het niet voor elke schrijver even goed, en ook niet onder elke omstandigheid.

    In 2019 stelden Margaret Mitchell en acht anderen de model card voor: een kort document bij een getraind model. Dat document zegt waarvoor het model is en waarvoor niet, welke data erachter staan, en hoe goed het model het doet, niet alleen in totaal maar per groep en per omstandigheid. Voor een model dat lachende gezichten zoekt op foto's betekent dat bijvoorbeeld per leeftijdsgroep, en bij fel en bij zwak licht. Hugging Face toont zo'n card op de pagina van een model, en makers van grote modellen publiceren er een bij elk model. In deze les meet je de cijferlezer zoals een card het hoort te melden, en vul je aan wat de card weglaat.

    **Een voorbeeld uit het dagelijks leven.** De bijsluiter in een doosje medicijnen zegt waarvoor het medicijn is, wie het niet mag nemen, zoals kinderen onder de 12, welke bijwerkingen bekend zijn en hoe vaak elke bijwerking voorkomt, en bij wie het getest is. Met alleen "werkt bij 9 van de 10 mensen" weet je niet of jij bij de negen hoort.
    """)
    return


@app.cell(hide_code=True)
def predict_employees():
    guess_employees = L.guess(
        "employees",
        question="De 1.000 testcijfers achter de 929 op de card zijn allemaal geschreven door Amerikaanse scholieren. QMNIST heeft ook cijfers van medewerkers van de Amerikaanse overheid. Neem er 1.000 van, van schrijvers die evenmin een trainingscijfer van de lezer schreven. Hoeveel daarvan leest de lezer volgens jou goed?",
        options={
            "same": "Ongeveer 930: dezelfde lezer, op dezelfde soort beelden",
            "fewer": "Ongeveer 880: volwassenen schrijven vaak sneller en slordiger",
            "right": "Ongeveer 960: de cijfers van de scholieren zijn lastiger",
            "far": "Ongeveer 700: de lezer zag deze schrijvers nooit",
        },
        reveal="**Ongeveer 960: 960 van de 1.000.** Op de cijfers van de scholieren maakt de lezer 71 fouten, op de cijfers van de medewerkers 40: bijna twee keer zo veel bij de scholieren. Volgens de eigen toelichting van QMNIST gelden de cijfers van de scholieren als \"substantially more challenging\". Hier is het ene getal op de card het laagste van de twee, maar het had net zo goed het hoogste kunnen zijn, en de card had het je niet verteld. De lezer leerde van 6.000 cijfers, 3.000 van medewerkers en 3.000 van scholieren, dus ontbrak geen van beide groepen. De 3.000 cijfers van de medewerkers kwamen zelfs van 29 schrijvers, tegen 250 schrijvers bij de scholieren. Zo per groep meten heeft een naam, in *Het idee*.",
    )
    guess_employees
    return (guess_employees,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Wat een model card zegt.** Mitchell en anderen noemen negen delen. De negen delen komen neer op vijf vragen:

        - **Wat is het?** Wat erin gaat en wat eruit komt, wie het model maakte, welke versie het is, en de licentie.
        - **Waarvoor is het, en waarvoor niet?** Het **beoogde gebruik** (in het Engels: intended use): de taken en de gebruikers die de makers voor ogen hadden. En het gebruik waarvoor het model niet bedoeld is, zoals een beslissing die het niet alleen mag nemen.
        - **Welke data staan erachter?** Waarvan het model leerde, waarop het getest is, en wie er in de data staan.
        - **Hoe goed doet het het, en voor wie?** De metric over de hele test set, en per groep en per omstandigheid.
        - **Wat gaat er mis?** De grenzen die de getallen laten zien, en wat iemand moet nagaan voor gebruik.

        **Per groep en per omstandigheid.** Een accuracy over een hele test set is een gemiddelde, en een gemiddelde kan een groep verbergen waarvoor het model slecht werkt, zoals in [Wie staat er in de data, en wie ontbreekt?](/course/open/rai-bias-in-data?lang=nl) **Disaggregated evaluation** meet de metric voor elke groep en voor elke omstandigheid apart. Een groep gaat over wie of wat er in de data staat: wie een cijfer schreef, of welk cijfer het is. Een omstandigheid gaat over hoe de invoer binnenkomt: een formulier dat scheef in de scanner ligt, of een cijfer dat niet in het midden van het vakje staat. Het beoogde gebruik bepaalt welke groepen en omstandigheden ertoe doen.
        """),
        mo.Html(C.split_picture(L.lang)),
        mo.md(r"""
        De cijferlezer op 2.000 testcijfers: de 1.000 cijfers van scholieren van de card en 1.000 cijfers van medewerkers. Elke balk toont het deel dat goed gelezen is in de kleur van de groep, en daarna het deel dat fout gelezen is in paars; de lijn staat op 90%, de lat die deze les legt voor het lezen van formulieren. Over alle 2.000 cijfers leest de lezer 94,5% goed. Per groep gaan de cijfers van de scholieren bijna twee keer zo vaak fout. En 15° met de klok mee gedraaid, zoals op een scheef liggend formulier, draait de volgorde om: dan gaan de cijfers van de medewerkers vaker fout. Een card met alleen de eerste balk verbergt allebei.

        **Kleine groepen.** Je kunt verder splitsen, tot elke schrijver apart. De 1.000 cijfers van scholieren komen van 247 schrijvers, van wie de meesten er 2 tot 6 schreven. Maar een getal over 2 cijfers kan alleen 0%, 50% of 100% zijn, en één fout verschuift het met de helft. Dus geeft een card bij elk getal hoeveel voorbeelden erachter staan, en laat de card groepen weg met te weinig voorbeelden om iets te zeggen.

        **Een system card.** Een model wordt zelden alleen gebruikt. In [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl) was het model een klein deel van een systeem. Een **system card** doet voor een heel systeem wat een model card voor één model doet: welke modellen het systeem gebruikt, wat er vooraf en daarna gebeurt, zoals een stap die elk cijfer uitknipt of een mens die controleert, en hoe goed het geheel het doet, getest zoals het gebruikt wordt. OpenAI publiceerde in 2023 een system card bij GPT-4: waarop experts het model testten, zoals schadelijke antwoorden, en welke maatregelen er rond het model kwamen.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **beoogd gebruik** (intended use) | | waarvoor een model is: de taken en de gebruikers die de makers voor ogen hadden; een card noemt ook gebruik waarvoor het model niet bedoeld is | |
        | **disaggregated evaluation** | | een metric meten voor elke groep en elke omstandigheid apart, niet alleen over de hele test set | `right[series == 4].float().mean()` |
        | **system card** | | een card voor een heel AI-systeem: de modellen, de stappen eromheen, en hoe goed het geheel het doet, getest zoals het gebruikt wordt | |
        """),
        L.watch_out("Een card meldt alleen de groepen en omstandigheden die iemand gemeten heeft. Een groep die niemand mat komt helemaal niet voor, en dat ziet eruit als goed nieuws. Dus zegt een card ook wat er niet getest is."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een model card schrijven:

    1. Zeg wat het model is: wat erin gaat, wat eruit komt, wie het maakte, de versie en de licentie.
    2. Schrijf het beoogde gebruik op, en het gebruik waarvoor het model niet bedoeld is.
    3. Zeg van welke data het model leerde en waarop het getest is, en wie erin staan.
    4. Kies de groepen en omstandigheden die ertoe doen voor het beoogde gebruik, en meet per groep en per omstandigheid, met het aantal voorbeelden achter elk getal.
    5. Schrijf de grenzen op: waar de getallen zakken onder wat het beoogde gebruik vraagt, en welke fouten bekend zijn.
    6. Is het model deel van een systeem? Dan krijgt het systeem ook een system card: de delen, en hoe goed het geheel het doet.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 4 voor één manier van splitsen: per cijfer. De cel laadt de cijferlezer zoals in [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl), en haalt de 2.000 testcijfers op met `C.test_digits()`: `images`, `digits`, het juiste cijfer bij elk beeld, en `series`, de serie van de schrijver van elk cijfer, zoals QMNIST het vastlegt. Serie 4 is geschreven door scholieren, serie 0 en 1 door medewerkers. `right` bevat True voor elk cijfer dat de lezer goed leest. Het mask `digits == _digit` houdt de cijfers van één soort over, en `_mask.sum()` telt hoeveel het er zijn: het aantal voorbeelden achter elk getal. Voer de cel uit.
    """)
    return


@app.cell
def example_digits():
    import torch
    from everything_ai.models import DigitCNN

    reader = DigitCNN()
    reader.load_state_dict(torch.load(C.hub_file("model.pt")[0], weights_only=True))
    reader.eval()
    images, digits, series = C.test_digits()
    with torch.no_grad():
        right = reader(images).argmax(dim=1) == digits
    # 4. in totaal, en per cijfer, met het aantal cijfers achter elk getal
    print("all", round(right.float().mean().item(), 3), len(right))
    for _digit in range(10):
        _mask = digits == _digit
        print(_digit, round(right[_mask].float().mean().item(), 3), _mask.sum().item())
    return digits, images, reader, series, torch


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Over alle 2.000 cijfers leest de lezer 0,945 goed. Per cijfer loopt dat van 0,89 voor een 8 tot 0,979 voor een 0: een 8 gaat ongeveer vijf keer zo vaak fout als een 0, 11,0 tegen 2,1 per 100. Elk van deze getallen staat op 164 tot 238 cijfers, genoeg om de getallen te vergelijken. Het beoogde gebruik bepaalt of deze splitsing op de card hoort: op een formulier telt elk cijfer van een bedrag.
    """)
    return


@app.cell(hide_code=True)
def text_conditions():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Per groep en per omstandigheid

    Het beoogde gebruik van de card in deze les: de cijfers lezen in de vakjes van gescande formulieren. Een formulier kan scheef in de scanner liggen, bij het uitknippen kan een vakje een paar pixels verschuiven, en iemand kan vergeten de cijfers wit op zwart te maken. De cel maakt de 2.000 cijfers onder vijf omstandigheden: zoals geschreven, 15° tegen de klok in en 15° met de klok mee gedraaid met `C.turn`, 3 pixels naar rechts geschoven met `C.move`, en donker op licht met `1 - images`. `right_when` bevat voor elke omstandigheid of de lezer elk cijfer goed leest. Voer de cel uit.
    """)
    return


@app.cell
def example_conditions(digits, images, reader, torch):
    changed = {
        "original": images,
        "turned +15": C.turn(images, 15),    # 15° tegen de klok in
        "turned -15": C.turn(images, -15),   # 15° met de klok mee
        "moved 3": C.move(images, 3),        # 3 pixels naar rechts
        "inverted": 1 - images,              # donker op licht
    }
    right_when = {}
    for _condition in changed:
        with torch.no_grad():
            right_when[_condition] = reader(changed[_condition]).argmax(dim=1) == digits
    list(right_when)
    return (right_when,)


@app.cell(hide_code=True)
def text_exercise_per_group():
    mo.vstack([
        mo.md(r"""
        Nu de tabel van de card. De startcode gaat de omstandigheden in `right_when` langs: voor elke `condition` bevat `right` of de lezer elk cijfer daaronder goed leest, en `table[condition]` krijgt de accuracy voor de scholieren (`"students"`) en voor de medewerkers (`"employees"`). Het mask `students = series == 4` houdt de cijfers van de scholieren over; `.float()` maakt van True en False 1 en 0, `.mean()` neemt het deel, en `.item()` maakt er een Python-getal van. Vervang de `...` op de regel voor de medewerkers: de cijfers van serie 0 en 1. De laatste regel is `table`, zodat de check de tabel ziet. De check zet de 2.000 cijfers ook in een andere volgorde.
        """),
        L.task("Maak de regel voor de medewerkers af, zodat `table` de accuracy van beide groepen onder elke omstandigheid bevat."),
    ])
    return


@app.cell(hide_code=True)
def exercise_per_group():
    box_per_group = L.code_box("per-group", lines=10)
    box_per_group
    return (box_per_group,)


@app.cell(hide_code=True)
def check_per_group(box_per_group, right_when, series):
    result_per_group = L.check("per-group", box_per_group.value, {"right_when": right_when, "series": series})
    result_per_group
    return (result_per_group,)


@app.cell(hide_code=True)
def hints_per_group():
    L.hints(
        "per-group",
        [
            "Stap 4 van het recept: hetzelfde als de regel voor de scholieren, `right[...].float().mean().item()`, met een mask uit `series` dat alleen de cijfers van de medewerkers overhoudt.",
            "Serie 0 en 1 zijn de medewerkers, serie 4 de scholieren. Welke ene vergelijking van `series` met 4 is True voor 0 en voor 1, en False voor 4?",
            "`\"employees\": right[series ___ 4].float().mean().item(),`",
        ],
        why="`series != 4` is True voor elk cijfer van serie 0 en van serie 1: alle 1.000 cijfers van de medewerkers. `series < 4` geeft hetzelfde mask. Zoals geschreven leest de lezer 0,929 van de cijfers van de scholieren goed en 0,96 van de medewerkers. 15° met de klok mee gedraaid draait de volgorde om: 0,86 tegen 0,827. Geschoven met 3 pixels, of donker op licht, zakken beide groepen ver onder 90%.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Verder draaien

    De tabel heeft 15° in elke richting. Met de slider leest de lezer alle 2.000 cijfers verder gedraaid, van 45° met de klok mee tot 45° tegen de klok in, in stappen van 5°. Bovenaan zes van de cijfers zoals de lezer de cijfers krijgt: drie van scholieren, drie van medewerkers.
    """)
    return


@app.cell(hide_code=True)
def demo_turn():
    _out = L.explore("turn", "graden", lambda _d: C.turn_frame(_d, L.lang), start=0, low=-45, high=45, step=5,
                     role="",
                     task="Draai beide kanten op. Waar zakt elke groep onder 90%? En in welke richting lijden de cijfers van de medewerkers het meest?")
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Binnen 10° naar elke kant blijven beide groepen boven 90%. Tegen de klok in houden de cijfers van de medewerkers de voorsprong: bij 30° leest de lezer er 73,6% van goed, tegen 64,7% van de cijfers van de scholieren. Met de klok mee zakken de cijfers van de medewerkers sneller: bij 30° nog 45,3% goed, tegen 63,3%. Een card kan daar een grens van maken waar iemand mee verder kan: leg elk formulier tot binnen 10° recht, voordat de lezer het formulier ziet.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sections():
    mo.md(r"""
    ### Oefening 2. De delen van de card

    Acht regels voor de card van de cijferlezer. Zet elke regel in het deel van de card waar de regel hoort: waarvoor het model is, waarvoor het niet is, waarvan het leerde en waarop het getest is, of de bekende grenzen.
    """)
    return


@app.cell(hide_code=True)
def sort_sections_cell():
    sort_sections = L.sort("sections")
    sort_sections
    return (sort_sections,)


@app.cell(hide_code=True)
def check_sections(sort_sections):
    result_sections = L.check("sections", sort_sections)
    result_sections
    return (result_sections,)


@app.cell(hide_code=True)
def hints_sections():
    L.hints(
        "sections",
        [
            "Vraag bij elke regel: is het een gebruik dat de makers voor ogen hebben, een gebruik waarvoor de makers waarschuwen, een feit over de cijfers achter de lezer, of een omstandigheid waaronder de lezer faalt?",
            "Twee regels zijn gebruik dat bij de lezer past, en twee zijn gebruik dat er niet bij past. Welke twee zeggen waar de getallen vandaan komen?",
            "Waarvoor het is: formulieren en \\_\\_\\_. Waarvoor het niet is: geld en \\_\\_\\_. Data: \\_\\_\\_ en testcijfers. Bekende grenzen: gedraaid en \\_\\_\\_.",
        ],
        why="**Waarvoor het is:** formulieren en lessen. **Waarvoor het niet is:** geld, omdat ongeveer 7 van de 100 cijfers van scholieren fout gaan, en straatfoto's, die in niets lijken op witte cijfers op zwart. **Waarvan het leerde en waarop het getest is:** de trainingscijfers en de testcijfers. **Bekende grenzen:** gedraaid en donker op licht, uit de getallen van oefening 1.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_system():
    mo.md(r"""
    ### Oefening 3. Een system card
    """)
    return


@app.cell(hide_code=True)
def choice_system_cell():
    choice_system = L.choice(
        "system-card",
        question="De app van een bank leest bedragen op formulieren: een foto van het formulier gaat erin, een stap knipt elk cijfer uit, de cijferlezer leest de cijfers, en als de lezer twijfelt, kijkt een mens mee. Wat hoort in de system card van de app, en niet in de model card van de lezer?",
        options={
            "groups": "De accuracy van de lezer per groep schrijvers en per omstandigheid, zoals in oefening 1",
            "license": "De licentie van de cijferlezer, en van de cijfers uit QMNIST waarvan de lezer leerde",
            "code": "De code van de class `DigitCNN`, zodat iedereen de lezer los kan laden",
            "right": "Hoe vaak de hele app een bedrag goed leest, en hoe vaak er een mens moet meekijken",
        },
        correct="right",
        why={
            "right": "Een system card gaat over het geheel: de foto, het uitknippen, de lezer en de mens samen. Een verkeerd uitgeknipt cijfer, of een bedrag waarin een van de zes cijfers fout gaat, zie je pas als de hele app getest wordt op echte formulieren.",
            "groups": "Dat gaat over de lezer alleen, dus hoort het op de model card van de lezer. Wat kun je alleen meten op de hele app, met de andere stappen erbij?",
            "license": "Dat gaat over de lezer en de data, dus hoort het op de model card van de lezer. Wat kun je alleen meten op de hele app, met de andere stappen erbij?",
            "code": "Dat gaat over de lezer alleen, dus hoort het op de model card van de lezer. Wat kun je alleen meten op de hele app, met de andere stappen erbij?",
        },
    )
    choice_system
    return (choice_system,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl) las je een model card; hier mat je een model zoals de card het hoort te melden: per groep schrijvers en per omstandigheid, met het aantal voorbeelden achter elk getal, en maakte je van deze getallen de grenzen. In *Project: tests en een model card* krijgt je model van de vraag zo'n card, met de getallen per groep uren en waarvoor het model niet is. *Een model card voor je model* schrijft de card van een eigen chatbot, met een header die Hugging Face leest.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_system, result_per_group, result_sections):
    L.wrap_up(result_per_group, result_sections, choice_system)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
