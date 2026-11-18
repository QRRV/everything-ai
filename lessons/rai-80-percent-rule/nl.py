# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-80-percent-rule. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-80-percent-rule", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Directe en indirecte discriminatie](/course/open/rai-discrimination-law?lang=nl) en [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_indirect():
    quick_indirect = L.choice(
        "quick-indirect",
        question="Een regel noemt geen discriminatiegrond, maar raakt vrouwen veel harder dan mannen. Hoe heet dat?",
        options={
            "direct": "Directe discriminatie, want het effect is hetzelfde",
            "right": "Indirecte discriminatie",
            "none": "Geen discriminatie, zolang niemand het zo bedoelde",
        },
        correct="right",
        why={
            "direct": "Directe discriminatie is als de regel zelf mensen slechter behandelt vanwege de grond. Zie *Het idee* in [Directe en indirecte discriminatie](/course/open/rai-discrimination-law?lang=nl).",
            "right": "Een neutrale regel die een groep in het bijzonder benadeelt, is indirecte discriminatie; die mag alleen als hij objectief gerechtvaardigd is.",
            "none": "Indirecte discriminatie vraagt geen opzet: het effect telt. Zie *Het idee* in [Directe en indirecte discriminatie](/course/open/rai-discrimination-law?lang=nl).",
        },
    )
    quick_indirect
    return (quick_indirect,)


@app.cell(hide_code=True)
def quick_check_groupby():
    quick_groupby = L.choice(
        "quick-groupby",
        question="`df` is een tabel met mensen en een kolom `sex`. Wat geeft `df.groupby(\"sex\").size()`?",
        options={
            "total": "Het aantal rijen van de hele tabel, als één getal",
            "right": "Hoeveel rijen elke waarde van `sex` heeft",
            "sorted": "De rijen van de tabel, gesorteerd op hun waarde van `sex`",
        },
        correct="right",
        why={
            "total": "`groupby` deelt de rijen eerst op in groepen, dus je krijgt één getal per groep. Zie *Het idee* in [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl).",
            "right": "Eén getal per groep: de rijen met die waarde van `sex`.",
            "sorted": "`groupby` vat elke groep samen in één getal. Zie *Het idee* in [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl).",
        },
    )
    quick_groupby
    return (quick_groupby,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Indirecte discriminatie zie je door te tellen tegen wie een regel ja zegt. Maar hoe groot moet een verschil tussen groepen zijn voordat het een signaal is? In de Verenigde Staten beantwoordt een vuistregel dat sinds 1978, en veel werk over fairness in machine learning bouwt erop voort. In Europa is er geen vast getal.

    **Een voorbeeld uit het dagelijks leven.** 40 meisjes en 40 jongens doen mee aan de selectie voor een sportteam; 12 meisjes en 24 jongens worden gekozen. Dat is 30 procent tegen 60 procent: de meisjes krijgen half zo vaak een ja als de jongens.
    """)
    return


@app.cell(hide_code=True)
def predict_women():
    guess_women = L.guess(
        "women",
        question="In een Amerikaanse tabel uit 1994 (UCI Adult) verdient 30,4 procent van de mannen meer dan 50.000 dollar per jaar. Wat gok je voor de vrouwen?",
        options={
            "same": "Ongeveer 28 procent, bijna even vaak als mannen",
            "two": "Ongeveer 20 procent",
            "right": "Ongeveer 11 procent, een derde zo vaak",
        },
        reveal="**Ongeveer 11 procent**: 10,9. Een model dat leert deze labels na te doen, geeft vrouwen ongeveer een derde zo vaak een ja als mannen.",
    )
    guess_women
    return (guess_women,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Het selectiepercentage.** Per groep het aantal mensen dat een ja krijgt, gedeeld door het aantal mensen in die groep: aangenomen, een lening, een plek in het team.

        **De 80%-regel.** In 1978 publiceerden vier Amerikaanse federale instanties, waaronder de EEOC, richtlijnen voor het selecteren van personeel. Eén regel daarin: een selectiepercentage van een ras, geslacht of etnische groep dat lager is dan vier vijfde, of 80 procent, van het percentage van de groep met het hoogste percentage, geldt in het algemeen als bewijs van adverse impact. Hun vragen en antwoorden uit 1979 geven vier stappen: bereken het percentage van elke groep, zoek het hoogste, deel elk percentage door het hoogste, en kijk of een verhouding onder 0,8 ligt. Hun voorbeeld: van 80 witte sollicitanten worden er 48 aangenomen, 60 procent, en van 40 zwarte sollicitanten 12, 30 procent. 30 gedeeld door 60 is 0,5: een teken van adverse impact.

        **Disparate impact.** Het Amerikaanse idee achter de regel. In 1971 oordeelde het Hooggerechtshof in *Griggs v. Duke Power* dat de wet werkwijzen verbiedt die "fair zijn van vorm, maar discriminerend in hun werking", tenzij ze met het werk te maken hebben. Duke Power eiste een middelbareschooldiploma of een gehaalde algemene intelligentietest; in een EEOC-zaak met dezelfde tests haalde 58 procent van de witte sollicitanten die, tegen 6 procent van de zwarte, een verhouding van 0,10. In 1991 werd disparate impact in de wet opgenomen. Sinds april 2025 zegt een presidentieel besluit tegen Amerikaanse instanties dat ze disparate impact zo veel mogelijk moeten laten vallen; het besluit verandert de wet zelf niet.

        **Een vuistregel.** De instanties schrijven dat de regel "niet bedoeld is als juridische definitie", maar als praktisch middel om de aandacht te houden bij ernstige verschillen. Bij kleine aantallen zegt hij weinig: 3 van 20 mannen en 1 van 10 vrouwen geeft 0,67, te weinig om adverse impact vast te stellen. En een kleiner verschil kan toch tellen als het statistisch en praktisch betekenisvol is.

        **In Europa.** Het Europese recht heeft geen vast getal. In 1999 vroeg het Hof van Justitie of een "aanzienlijk kleiner percentage" vrouwen dan mannen aan een regel kon voldoen; bij 77,4 procent van de mannen en 68,9 procent van de vrouwen leken de cijfers dat volgens het Hof op het eerste gezicht niet te laten zien. De verhouding is 0,89, dus de 80%-regel zou ook geen signaal geven, maar het Hof gebruikte geen verhouding. In Nederland vond het College voor de Rechten van de Mens het genoeg voor een vermoeden van indirecte discriminatie dat een werkgever na een assessment alle mannen doorliet en maar een op de drie vrouwen.

        **De tabel UCI Adult.** Deze tabel bevat 48.842 mensen uit een Amerikaans onderzoek van 1994, met een label: verdient meer dan 50.000 dollar per jaar of niet. Hij is gebruikt in meer dan driehonderd papers over fairness. Die grens ligt hoog: in 1994 lag hij boven 89 procent van de inkomens van vrouwen. Een model dat de labels nadoet, geeft vrouwen ongeveer een derde zo vaak een ja als mannen.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **selectiepercentage** | het aandeel van een groep dat een ja krijgt: de geselecteerden uit de groep gedeeld door de mensen in de groep |
        | **80%-regel** | een vuistregel: een groep waarvan het selectiepercentage onder 80 procent van het hoogste ligt, is een signaal; in het Engels de *four-fifths rule* |
        | **disparate impact** | in het Amerikaanse recht een werkwijze die neutraal lijkt maar een beschermde groep veel meer treft dan anderen; onrechtmatig, tenzij die met het werk te maken heeft |
        | **adverse impact** | het woord van de richtlijnen voor een duidelijk lager selectiepercentage dat nadelig uitvalt voor een ras, geslacht of etnische groep |
        | **impact ratio** | de verhouding: het selectiepercentage van een groep gedeeld door het hoogste selectiepercentage; onder 0,8 is het signaal van de regel |
        """),
        L.watch_out("Slagen voor de regel laat niet zien dat een proces fair is, en zakken bewijst geen discriminatie. Het is een eerste signaal dat vraagt om beter te kijken: naar de aantallen erachter, en naar de vraag of de regel gerechtvaardigd is."),
        mo.Html(C.picture({"title": "Hoe vaak de labels uit UCI Adult ja zeggen",
                           "groups": {"Female": "vrouwen", "Male": "mannen"},
                           "ylabel": "selectiepercentage", "line": "vier vijfde van het hoogste",
                           "decimal": ",",
                           "alt": "Twee staven: het aandeel met een ja volgens de labels is 0,109 bij vrouwen en 0,304 bij mannen. Een gestippelde lijn op vier vijfde van het hoogste percentage, 0,243, ligt ver boven de staaf van de vrouwen."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo pas je de 80%-regel toe op een model of een regel:

    1. Bepaal wat als ja telt: aangenomen, een lening, verdient meer dan 50.000 dollar.
    2. Deel per groep de mensen met een ja door alle mensen: het selectiepercentage.
    3. Zoek het hoogste selectiepercentage.
    4. Deel elk percentage door het hoogste: de verhouding.
    5. Onder 0,8 is een signaal. Kijk dan op hoeveel mensen de percentages rusten, en vraag of de regel gerechtvaardigd is.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De labels van UCI Adult, per geslacht. `_picked` bevat de rijen met een ja: een inkomen boven 50.000 dollar. `_rates` deelt de gekozen mensen per geslacht door alle mensen per geslacht: `groupby("sex").size()` telt de rijen van elke groep, en de twee tellingen worden groep voor groep gedeeld. `_ratios` deelt elk percentage door het hoogste, `_rates.max()`. Voer de cel uit.
    """)
    return


@app.cell
def example_ratios():
    import pandas as pd

    # 48.842 mensen uit het Amerikaanse onderzoek van 1994 (UCI Adult)
    _people = pd.read_csv("data/adult.csv.gz")

    _picked = _people[_people["income"] == ">50K"]
    _rates = _picked.groupby("sex").size() / _people.groupby("sex").size()
    _ratios = _rates / _rates.max()
    print(_rates.round(3))
    print(_ratios.round(3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De vrouwen krijgen in 0,109 van de gevallen een ja, de mannen in 0,304: een verhouding van 0,36, ver onder 0,8. Python schrijft getallen met een punt. Nu een regel die helemaal geen geslacht noemt: Oefening 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ratios():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een regel over werkuren

        Een regel die de course verzon: ja voor wie minstens `hours` uur per week werkt, 45. Vervang de `...` door de drie regels van het uitgewerkte voorbeeld, met de namen zonder liggend streepje: `picked`, `rates` en `ratios`. `picked` bevat de rijen waarin `hours_per_week` minstens `hours` is. De laatste regel is `ratios`, zodat de check het ziet. De check probeert ook een ander aantal uren.
        """),
        L.task("Reken `ratios` uit: de verhouding van vrouwen en van mannen bij de regel over uren."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ratios():
    box_ratios = L.code_box("ratios", lines=11)
    box_ratios
    return (box_ratios,)


@app.cell(hide_code=True)
def check_ratios(box_ratios):
    result_ratios = L.check("ratios", box_ratios.value)
    result_ratios
    return (result_ratios,)


@app.cell(hide_code=True)
def hints_ratios():
    L.hints(
        "ratios",
        [
            "Het uitgewerkte voorbeeld kiest de rijen met een hoog inkomen. Hier kies je de rijen met genoeg uren, en de andere twee regels blijven hetzelfde.",
            "Kies met een vergelijking tussen de haken: de kolom `hours_per_week`, minstens `hours`. Dan de percentages per geslacht, en deel die door het hoogste.",
            "De eerste regel: `picked = people[people[\"hours_per_week\"] >= ___]`; de laatste: `ratios = rates / ___`.",
        ],
        why="`picked = people[people[\"hours_per_week\"] >= hours]`, dan `rates = picked.groupby(\"sex\").size() / people.groupby(\"sex\").size()` en `ratios = rates / rates.max()`. Dat geeft ongeveer `0.433` voor vrouwen en `1.0` voor mannen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_flag():
    mo.md(r"""
    ### Oefening 2. Welke groepen geeft de regel aan?
    """)
    return


@app.cell(hide_code=True)
def choice_flag_cell():
    choice_flag = L.choice(
        "flag",
        question="In een ander voorbeeld van de Amerikaanse instanties is het percentage aangenomen sollicitanten 60 procent bij witte sollicitanten, 45 procent bij inheemse Amerikanen, 48 procent bij Latino's en 51 procent bij zwarte sollicitanten. Bij welke groepen geeft de 80%-regel een signaal van adverse impact?",
        options={
            "right": "Alleen bij de groep met 45 procent",
            "two": "Bij de groepen met 45 en 48 procent",
            "three": "Bij alle drie de groepen",
        },
        correct="right",
        why={
            "right": "45 gedeeld door 60 is 0,75, onder 0,8. 48 gedeeld door 60 is precies 0,80: niet minder dan vier vijfde. 51 gedeeld door 60 is 0,85. De instanties komen op hetzelfde uit.",
            "two": "48 gedeeld door 60 is precies 0,80. De regel geeft een signaal bij een percentage dat *minder dan* vier vijfde van het hoogste is. Deel elk percentage door 60.",
            "three": "Deel elk percentage door het hoogste, 60, en vergelijk met 0,8: 51 gedeeld door 60 is 0,85.",
        },
    )
    choice_flag
    return (choice_flag,)


@app.cell(hide_code=True)
def text_exercise_meaning():
    mo.md(r"""
    ### Oefening 3. Wat zegt het als je erdoor komt?
    """)
    return


@app.cell(hide_code=True)
def choice_meaning_cell():
    choice_meaning = L.choice(
        "meaning",
        question="Een wervingsprogramma geeft elke groep minstens 80 procent van het hoogste selectiepercentage. Wat zegt dat?",
        options={
            "fair": "Het programma is fair",
            "right": "Alleen dat één grof signaal ontbreekt",
            "law": "Het voldoet aan het Nederlandse en Europese recht, dat dezelfde regel gebruikt",
        },
        correct="right",
        why={
            "fair": "De instanties schrijven dat de regel niet beslist of er onrechtmatig wordt gediscrimineerd, en dat kleinere verschillen toch kunnen tellen. Zie *Een vuistregel* in *Het idee*.",
            "right": "De regel houdt alleen de aandacht bij ernstige verschillen. Een kleiner verschil kan toch tellen, en bij kleine aantallen zegt de regel weinig, de ene kant op of de andere.",
            "law": "Het Europese en Nederlandse recht hebben geen vast getal: het Hof van Justitie vraagt of een aanzienlijk kleiner percentage van een groep aan een regel kan voldoen. Zie *In Europa* in *Het idee*.",
        },
    )
    choice_meaning
    return (choice_meaning,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Directe en indirecte discriminatie](/course/open/rai-discrimination-law?lang=nl) zag je dat een neutrale regel indirecte discriminatie is als hij een groep in het bijzonder treft; de verhouding is een manier om een getal te zetten op *in het bijzonder*. Het tellen is de `groupby` uit [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl): één telling per groep, groep voor groep gedeeld. De volgende les, *Waarom is discriminatie fout?*, vraagt wat zo'n verschil eigenlijk fout maakt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_flag, choice_meaning, result_ratios):
    L.wrap_up(result_ratios, choice_flag, choice_meaning)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
