# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les start-using-ai-tools. Route: course/route/lessen/stam-1-basis.yaml. Checks en de verzonnen volgende woorden: checks.py.
# termen:allow genai

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("start-using-ai-tools", "nl")
    C = L.checks   # de verzonnen volgende woorden voor het beeld


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over de vier cirkels.
    """)
    return


@app.cell(hide_code=True)
def quick_check_sketch():
    quick_sketch = L.choice(
        "quick-sketch",
        question="Een app maakt van je ruwe schets een compleet schilderij, met een neural network dat van heel veel schilderijen heeft geleerd. Wat is de kleinste cirkel waar hij in past?",
        options={"ml": "Machine learning", "dl": "Deep learning", "genai": "Generatieve AI"},
        correct="genai",
        why={
            "ml": "Hij past in machine learning, maar er zijn kleinere cirkels: er zit een neural network in, en hij maakt iets. Loop de drie vragen uit [AI, machine learning, deep learning en generatieve AI](/course/open/start-ai-ml-genai?lang=nl) nog eens langs.",
            "dl": "Hij past in deep learning, maar er is een kleinere cirkel. Maakt hij iets nieuws, of sorteert en voorspelt hij alleen? Kijk nog eens in [AI, machine learning, deep learning en generatieve AI](/course/open/start-ai-ml-genai?lang=nl).",
            "genai": "Hij heeft van voorbeelden geleerd, met een neural network, en hij maakt iets nieuws: een schilderij. Dus is het generatieve AI.",
        },
    )
    quick_sketch
    return (quick_sketch,)


@app.cell(hide_code=True)
def quick_check_question():
    quick_question = L.choice(
        "quick-question",
        question="Met welke vraag ontdek je of een systeem dat deep learning gebruikt, ook generatieve AI is?",
        options={
            "examples": "Heeft het van voorbeelden geleerd?",
            "layers": "Is het een neural network met veel layers?",
            "new": "Maakt het iets nieuws, zoals tekst of beelden?",
        },
        correct="new",
        why={
            "new": "Generatieve AI is het deel van deep learning dat iets nieuws maakt.",
            "examples": "Elk systeem met deep learning heeft van voorbeelden geleerd, dus met deze vraag houd je de twee niet uit elkaar. Kijk nog eens naar de laatste vraag van het recept in [AI, machine learning, deep learning en generatieve AI](/course/open/start-ai-ml-genai?lang=nl).",
            "layers": "Elk systeem met deep learning is een neural network met veel layers, dus met deze vraag houd je de twee niet uit elkaar. Kijk nog eens naar de laatste vraag van het recept in [AI, machine learning, deep learning en generatieve AI](/course/open/start-ai-ml-genai?lang=nl).",
        },
    )
    quick_question
    return (quick_question,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Chatprogramma's zoals ChatGPT, Claude en Gemini (bekende voorbeelden in oktober 2026) schrijven e-mails, leggen huiswerk uit, plannen reizen en schrijven code. In 2026 gebruiken honderden miljoenen mensen ze, en ook in deze course kunnen ze je helpen. Maar ze zitten er ook weleens naast, en een fout antwoord komt vaak zonder waarschuwing. Weet je hoe ze werken, dan weet je wanneer je op een antwoord kunt vertrouwen en wanneer je het beter controleert.

    **Een voorbeeld uit het dagelijks leven.** Stel je iemand voor die ontzettend veel gelezen heeft en elke vraag meteen beantwoordt, vlot en zonder te aarzelen, ook als het eigenlijk niet zeker is. Met een brief laat je je graag door zo iemand helpen. Maar neem je op dat advies een medicijn in, dan lees je eerst de bijsluiter. Dat is maar een vergelijking: een chatprogramma leest en twijfelt niet zoals een mens. Wat het in plaats daarvan doet, zie je hieronder.
    """)
    return


@app.cell(hide_code=True)
def predict_sound():
    guess_sound = L.guess(
        "how-it-sounds",
        question="Een chatprogramma geeft je een antwoord dat niet klopt. Hoe klinkt dat antwoord, denk je?",
        options={
            "unsure": "Minder zeker dan anders: het aarzelt.",
            "same": "Net zo zeker als een goed antwoord.",
            "warns": "Het waarschuwt dat het ernaast kan zitten.",
        },
        reveal="**Vaak net zo zeker als een goed antwoord.** Hoe zeker een antwoord klinkt, zegt weinig over hoe juist het is. In deze les zie je waarom, en wat je eraan kunt doen.",
    )
    guess_sound
    return (guess_sound,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **AI-assistent** | een chatprogramma dat vragen beantwoordt, dingen uitlegt, teksten opzet en code schrijft; dat heeft het geleerd van een enorme hoeveelheid tekst |

    Hierna zeggen we gewoon *assistent*.

    ### Woord voor woord

    Achter het chatvenster zit een model: het deel dat van een enorme hoeveelheid tekst heeft geleerd hoe tekst verdergaat. Het kijkt naar de tekst die er al staat: jouw vraag en het antwoord tot nu toe. Dan kiest het een waarschijnlijk volgend woord, zet het erachter, en doet hetzelfde opnieuw, woord voor woord, tot het antwoord af is. Zo'n model heet een language model. Aan het eind van de basis bouw je er zelf een kleintje, en dan zie je het idee van binnenuit.

    Het zoekt het antwoord niet op in een lijst met feiten: wat het geleerd heeft, is hoe tekst meestal verdergaat. Ook dit is een vereenvoudiging: het model werkt eigenlijk met stukjes van woorden, en een latere les laat zien hoe. Het punt blijft staan: **waarschijnlijk is niet hetzelfde als waar.** Bij een vraag die in veel teksten beantwoord wordt, klopt het waarschijnlijke antwoord meestal. Bij iets waar weinig over geschreven is, zoals een feit over je eigen dorp of een precies getal, of bij iets wat gebeurde nadat het model zijn tekst leerde, kan het toch een aannemelijk antwoord schrijven, en net zo vlot.
    """)
    return


@app.cell(hide_code=True)
def picture_next_words():
    mo.vstack([
        draw.next_word_bars(
            "De molen in ons dorp is gebouwd in", C.NEXT_WORDS["nl"], picked=0, picked_label="komt in het antwoord",
            title="Hoe waarschijnlijk elk volgend woord is"),
        mo.md(r"""
        Elke **blauwe balk** laat zien hoe waarschijnlijk het model een volgend woord vindt; de rest is verdeeld over heel veel andere woorden. De getallen zijn voor dit voorbeeld verzonnen, maar zo werkt het wel: een paar jaartallen klinken waarschijnlijk, het model schrijft er één op, en niets in het antwoord verraadt dat het gokte.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_good_at():
    mo.vstack([
        mo.md(r"""
        ### Waar een assistent goed in is, en waar je op moet letten

        **Goed in:** iets in andere woorden uitleggen, voorbeelden geven, een eerste versie van een tekst schrijven, ideeën bedenken, een tekst in een andere toon herschrijven, eenvoudige code schrijven.

        **Let op bij:** precieze feiten over zeldzame dingen; getallen, datums, citaten en bronnen; iets wat net gebeurd is; en alles waarbij een fout veel kost, zoals gezondheid, geld, juridische zaken, en code die iets verwijdert of verstuurt.

        De grens tussen die twee is niet scherp. Een assistent kan in één antwoord een moeilijk idee goed uitleggen en een simpele datum verkeerd hebben.
        """),
        L.watch_out("Een assistent kan meer dan het model dat erin zit: sommige assistenten zoeken op internet of lezen een bestand dat je ze geeft. Dat helpt, maar het language model schrijft het antwoord nog steeds woord voor woord, en het kan zeggen dat er iets op een pagina staat wat er niet op staat."),
        mo.md(r"""
        ### Recept: controleer wat ertoe doet

        1. **Wat ga je met het antwoord doen?** Gebruik je het alleen als idee of als eerste versie die je zelf doorloopt? Dan kun je het zo gebruiken. Geef je het door, ga je erop vertrouwen of ga je ernaar handelen? Ga dan door naar stap 2.
        2. **Waar hangt dat van af?** Haal de feiten eruit: namen, getallen, datums, citaten, bronnen, en wat een stuk code doet.
        3. **Controleer die delen ergens anders,** bij een bron die je vertrouwt: het oorspronkelijke document, een officiële website, een boek, of door de code uit te proberen op een kopie, waar een fout geen kwaad kan. Niet bij dezelfde assistent: die kan zijn eigen fout bevestigen.
        4. **Kon je het niet controleren?** Laat het weg, of zeg erbij dat het niet gecontroleerd is.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Je schrijft voor de schoolkrant een stukje over de molen in je dorp, en je vraagt een assistent om een begin. Dit antwoord is verzonnen, maar wel typisch:

    > Hier is een begin voor je stukje. De molen aan de Dijkweg is gebouwd in 1782 en is een van de oudste in de regio. Volgens de historische vereniging is hij twee keer verplaatst. Op een zomeravond is hij prachtig om te zien. Bezoekers zijn elke zaterdag welkom. Een titel kan zijn: *Na al die jaren draait hij nog steeds*.

    1. **Wat ga je ermee doen?** Je publiceert het, en lezers nemen het voor waar aan. Dus door. (Had je alleen om ideeën voor een titel gevraagd, dan was je hier klaar: ideeën waar je zelf uit kiest, hoef je niet te controleren.)
    2. **Waar hangt dat van af?** Van de feiten: gebouwd in 1782, een van de oudste in de regio, twee keer verplaatst volgens de historische vereniging, elke zaterdag open. Geen feiten: "prachtig om te zien" is een mening die je houdt of schrapt, en de titel kies je zelf.
    3. **Controleer ze ergens anders.** Het bouwjaar en of hij verplaatst is: op de website van de historische vereniging, of in het gemeentearchief. De openingsdagen op de eigen website van de molen, of op het bordje bij de deur. "Een van de oudste" door te vergelijken met een lijst van de molens in de regio.
    4. **Kon je het niet controleren?** Stel dat de website van de historische vereniging niets zegt over verplaatsen. Dan gaat die zin eruit, tot je een bron vindt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_check():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Welke delen zijn feiten?

        Je plant een dagje museum met je oma, die in een rolstoel zit, en je vraagt een assistent om hulp. Zijn antwoord bestaat uit de zes delen hieronder. Stap 1 van het recept is al gedaan: op de dag zelf ga je op dit plan vertrouwen. Doe nu stap 2.
        """),
        L.task("Zet elk deel van het antwoord in de goede groep."),
    ])
    return


@app.cell(hide_code=True)
def exercise_check():
    sort_check = L.sort("what-to-check")
    sort_check
    return (sort_check,)


@app.cell(hide_code=True)
def check_what_to_check(sort_check):
    result_check = L.check("what-to-check", sort_check)
    result_check
    return (result_check,)


@app.cell(hide_code=True)
def hints_check():
    L.hints(
        "what-to-check",
        [
            "Stap 2 van het recept: haal de feiten eruit. Namen, getallen, tijden, prijzen en wat er op de dag zelf kan.",
            "Twee delen zijn helemaal geen feiten: het zijn voorstellen, net als de titel in het uitgewerkte voorbeeld.",
            "Vraag je bij elk deel af: gaat de dag mis als dit niet klopt? Bij vier delen is dat zo.",
        ],
        why="Een feit waar de dag van afhangt: **Openingstijden**, **Rolstoel**, **Kaartjes** en **Trein**. Geen feit: **Tip** en **Idee**, voorstellen waar je zelf over beslist.",
    )
    return


@app.cell(hide_code=True)
def text_how_check():
    mo.md(r"""
    ### Oefening 2. Hoe controleer je?
    """)
    return


@app.cell(hide_code=True)
def choice_how_check_cell():
    choice_how_check = L.choice(
        "how-check",
        question="Je wilt zeker weten dat elke zaal van het museum met een lift te bereiken is. Hoe controleer je dat het best?",
        options={
            "again": "Je vraagt dezelfde assistent: \"Weet je het zeker van die lift?\"",
            "source": "Je kijkt op de website van het museum zelf, of je belt het museum.",
            "link": "Je vraagt de assistent om een link, en je vertrouwt het antwoord omdat er een link bij staat.",
        },
        correct="source",
        why={
            "again": "Dezelfde assistent is geen tweede bron: hij kan zijn fout herhalen, of juist een ander antwoord geven, en klinkt in beide gevallen even zeker. Stap 3 van het recept: wie weet dit zeker?",
            "source": "Het museum kent zijn eigen gebouw. Zo controleer je bij de bron, stap 3 van het recept.",
            "link": "Een link alleen bewijst niets: op de pagina kan iets anders staan, of de pagina bestaat helemaal niet. Stap 3 van het recept: wie weet dit zeker?",
        },
    )
    choice_how_check
    return (choice_how_check,)


@app.cell(hide_code=True)
def text_explain():
    mo.md(r"""
    ### Oefening 3. Waarom zo zeker?

    Schrijf eerst je eigen antwoord. Het modelantwoord waarmee je het daarna vergelijkt, is een voorbeeldantwoord van de course, geen antwoord van een AI-model.
    """)
    return


@app.cell(hide_code=True)
def explain_why_sure():
    explain_sure = L.explain(
        "why-sure",
        question="Waarom kan een assistent een verkeerd jaartal net zo vlot opschrijven als een goed jaartal? Schrijf een of twee zinnen.",
        model_answer="Het language model in de assistent schrijft het antwoord woord voor woord, en kiest steeds een woord dat waarschijnlijk volgt op de tekst tot dan toe. Een jaartal dat waarschijnlijk klinkt, is niet altijd het goede jaartal, en een waarschijnlijk woord kiezen is iets anders dan een feit controleren. Daarom komt een verkeerd jaartal er net zo vlot uit.",
        points=[
            "het kiest woord voor woord een waarschijnlijk volgend woord",
            "waarschijnlijk is niet hetzelfde als waar",
            "een waarschijnlijk woord kiezen is iets anders dan een feit controleren",
        ],
    )
    explain_sure
    return (explain_sure,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [AI, machine learning, deep learning en generatieve AI](/course/open/start-ai-ml-genai?lang=nl) zette je zo'n chatprogramma in de kleinste cirkel: generatieve AI. Nu weet je wat er in een AI-assistent zit: een language model dat woord voor woord schrijft. Aan het eind van de basis bouw je zelf een klein language model, en dan zie je waar "waarschijnlijk maar fout" vandaan komt; een latere les geeft dat een eigen naam. Het recept uit deze les blijft de hele course bruikbaar, ook als een assistent je helpt met je eigen code.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_how_check, explain_sure, result_check):
    L.wrap_up(result_check, choice_how_check, explain_sure)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
