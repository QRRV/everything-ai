# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les start-prompt-chains. Route: course/route/lessen/bouwen.yaml. Checks, de vier experimenten en de run van de course: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("start-prompt-chains", "nl")
    C = L.checks   # de vier experimenten, hoe de les een antwoord leest, en de run van Qwen3 van de course
    R = C.course()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_delimit():
    quick_delimit = L.choice(
        "quick-delimit",
        question="Je plakt de e-mail van een klant in een prompt, en vraagt een model hem samen te vatten. Waarom zet je de e-mail tussen `<email>` en `</email>`?",
        options={
            "speed": "Zodat het model hem sneller leest",
            "apart": "Zodat het model je instructie en de e-mail uit elkaar kan houden",
            "needed": "Omdat een model een e-mail zonder tags niet kan lezen",
        },
        correct="apart",
        why={
            "speed": "Tags kosten een paar tokens extra; sneller lezen wordt het er niet van. Wat als er in de e-mail zelf staat *Antwoord vandaag nog*? Zie [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl).",
            "apart": "De tags zijn scheidingstekens: een zin in de e-mail, zoals *Antwoord vandaag nog*, leest het model als deel van de tekst, niet als een instructie.",
            "needed": "Zonder tags leest het model de e-mail ook; het ziet alleen minder goed waar je instructie ophoudt. Zie [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl).",
        },
    )
    quick_delimit
    return (quick_delimit,)


@app.cell(hide_code=True)
def quick_check_last_line():
    quick_last_line = L.choice(
        "quick-last-line",
        question="Je wilt dat een model een som uitrekent, en een programma leest de uitkomst. Hoe eindig je de prompt?",
        options={
            "only": "*Geef alleen het getal.*",
            "first": "*Geef eerst het antwoord, dan je stappen.*",
            "last": "*Reken het stap voor stap uit. Geef dan het eindantwoord op de laatste regel, als: Answer: &lt;getal&gt;*",
        },
        correct="last",
        why={
            "only": "Met alleen het getal heeft het model geen ruimte om de som uit te rekenen voordat het het antwoord schrijft. Zie de tabel van de zeven manieren in [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl).",
            "first": "Als het antwoord er eenmaal staat, kunnen de stappen die erna komen het niet meer veranderen. Zie [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl).",
            "last": "Eerst de stappen, waarin het model het uitrekent, en dan het antwoord op een vaste laatste regel, waar een programma het kan vinden.",
        },
    )
    quick_last_line
    return (quick_last_line,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Het antwoord van een model leest goed, of het nu klopt of niet. En één prompt die veel tegelijk vraagt, geeft het model veel plekken om de fout in te gaan, zonder je te laten zien waar. Drie gewoonten helpen. Deel een grote taak op in kleine prompts, de ene na de andere, zodat je onderweg naar elk antwoord kunt kijken. Laat het model zijn eigen antwoord nakijken, of controleer elke bewering met een eigen vraag. En geef het model de tekst om uit te antwoorden, met het verzoek te noemen waar het elk feit vond, zodat je het kunt opzoeken. Deze les probeert elke gewoonte uit op Qwen3-0.6B, het kleine chat model uit [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl), elk acht keer, en laat zien wat elke gewoonte oplost, en wat niet.
    """)
    return


@app.cell(hide_code=True)
def predict_refine():
    guess_refine = L.guess(
        "refine",
        question=C.refine_question(L.lang),
        options={
            "most": "De meeste",
            "some": "Een paar",
            "none": "Geen enkele",
        },
        reveal=C.refine_reveal(L.lang),
    )
    guess_refine
    return (guess_refine,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Prompt chaining** deelt een taak op in stappen, elk een eigen prompt: het antwoord van de ene gaat in de volgende. Elke prompt vraagt minder, en tussen twee stappen kun jij, of je programma, naar het antwoord tot nu toe kijken.

    Een model kan ook naar zijn eigen antwoord kijken. **Self-Refine** vraagt het, in een tweede prompt, zijn antwoord na te kijken en te verbeteren. **Chain-of-Verification** gaat verder: eerst een eerste versie van het antwoord; dan, voor elke bewering daarin, een losse vraag in een nieuwe prompt, zonder die eerste versie, zodat het model het niet gewoon eens is met wat het schreef; en dan een eindantwoord met alleen de beweringen die de controle doorstonden.

    Een model kan ook antwoorden uit een tekst die je het geeft, in plaats van uit wat het onthouden heeft: dat is **grounding**. Vraag het de zin te noemen waar elk deel van zijn antwoord vandaan komt, een **citatie**, en je kijkt het antwoord in een oogwenk na in de tekst. Vraag het ook om het te zeggen als de tekst geen antwoord op de vraag geeft.

    Een prompt kan meer bevatten dan tekst. Een **multimodale prompt** heeft er ook een plaatje, een geluid of een bestand in, voor een model dat die kan lezen: een foto van een kassabon, met de vraag wat alles bij elkaar kostte. Qwen3-0.6B leest alleen tekst, dus deze les houdt het bij tekst.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | Voorbeeld |
    |---|---|---|
    | **prompt chaining** | een taak opdelen in kleine prompts na elkaar, waarbij het antwoord van elke prompt in de volgende gaat | eerst de hoeveelheden, dan de kosten |
    | **Self-Refine** | het model zijn eigen antwoord laten nakijken in een tweede prompt, en verbeteren | *Check your answer step by step. If it is wrong, fix it.* |
    | **Chain-of-Verification** | een eerste versie van het antwoord, dan een nieuwe vraag voor elke bewering erin, dan een eindantwoord met alleen de beweringen die de controle doorstonden | *Is 101 a prime number between 80 and 100?* |
    | **grounding** | een model laten antwoorden uit een tekst die je het geeft, niet uit wat het onthouden heeft | *Answer using only the text between the source tags.* |
    | **citatie** | de plek in de gegeven tekst waar een deel van het antwoord vandaan komt | *[2]* |
    | **multimodale prompt** | een prompt met een plaatje, een geluid of een bestand erin, naast tekst | een foto van een kassabon en *Wat kostte dit?* |
    """)
    return


@app.cell(hide_code=True)
def picture_habits():
    mo.vstack([
        draw.flow(
            [("Prompt 1", ["de hoeveelheden", f"voor {C.PEOPLE} mensen"]), ("Prompt 2", ["de hoeveelheden van", "antwoord 1, met de prijzen"]),
             ("Antwoord 2", ["de kosten, op", "de laatste regel"])],
            ["antwoord 1: de\nhoeveelheden", "antwoord 2"], ["", ""], title="Prompt chaining: één taak in twee prompts"),
        draw.flow(
            [("Eerste versie", ["vijf priemgetallen", "tussen 80 en 100"]), ("Eén vraag per getal", ["Is 89 priem en", "tussen 80 en 100? …", "elk in een nieuwe prompt"]),
             ("Het eindantwoord", ["alleen de getallen", "die doorkwamen"])],
            ["elk getal uit de\neerste versie", "ja of nee\nvoor elk"], ["", ""], title="Chain-of-Verification"),
        mo.md(r"""
        Tussen elke twee vakken staat een antwoord waar je naar kunt kijken: de hoeveelheden voordat ze een prijs krijgen, elk getal voordat het in de lijst komt. De vier experimenten hieronder zijn deze twee plaatjes, Self-Refine en grounding, op Qwen3-0.6B. De prompts zijn in het Engels, zoals in [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl), omdat Qwen3-0.6B het best Engels schrijft.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_chain():
    mo.md(rf"""
    ### Experiment 1. Eén prompt of een keten

    De taak, in het Engels: *{C.RECIPE} I make pancakes for {C.PEOPLE} people. {C.PRICES} How much do the ingredients cost?* Kort gezegd: wat kosten de ingrediënten als je het recept voor 4 mensen omrekent naar 12 mensen? Het antwoord is {C.COST} euro. In één prompt moet het model het recept tegelijk omrekenen en er een prijs aan geven. In een keten vraagt prompt 1 alleen om de hoeveelheden, en krijgt prompt 2 die hoeveelheden met de prijzen. Allebei vragen ze eerst om de stappen, en het antwoord op de laatste regel. De course deed allebei {R['seeds']} keer.
    """)
    return


@app.cell(hide_code=True)
def chain_cell():
    L.explore("chain-runs", "run", lambda _v: C.chain_frame(int(_v), L.lang), start=C.CHAIN_SHOWN, low=1, high=R["seeds"], step=1, role="",
              task="Loop de runs door. Als de keten de fout ingaat, in welke prompt gebeurt dat, en had je dat vóór prompt 2 kunnen zien?")
    return


@app.cell(hide_code=True)
def text_chain_after():
    mo.md(C.chain_after(L.lang))
    return


@app.cell(hide_code=True)
def text_refine():
    mo.md(rf"""
    ### Experiment 2. Zijn eigen antwoord nakijken

    De course nam de eerste antwoorden van Qwen3 op de drie sommen uit [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl), gevraagd zoals ze zijn en met *Reply with only the final number*, en gaf elk antwoord terug aan Qwen3 met nog één bericht: *{C.markdown_safe(C.CHECK)}* (controleer je antwoord stap voor stap, verbeter het als het fout is, en geef het eindantwoord op de laatste regel). De tabel telt de antwoorden die goed waren vóór de controle en erna.
    """)
    return


@app.cell(hide_code=True)
def text_refine_table():
    mo.md(C.refine_table(L.lang) + "\n\n" + C.refine_after(L.lang))
    return


@app.cell(hide_code=True)
def text_verify():
    mo.md(rf"""
    ### Experiment 3. Chain-of-Verification

    De eerste versie: *{C.draft_primes()[0]['content']}* (noem vijf priemgetallen tussen 80 en 100, alleen de lijst). Er zijn er maar drie tussen {C.LOW} en {C.HIGH}: {', '.join(str(n) for n in sorted(C.PRIMES)[:-1])} en {max(C.PRIMES)}, dus een eerste versie van vijf moet ergens fout zijn. Daarna, voor elk getal in de eerste versie, een nieuwe prompt, op twee manieren. Gewoon: *{C.verify(101)[0]['content']}* En eerst uitgerekend: *{C.verify_steps(101)[0]['content']}* De eindlijst houdt de getallen die een ja kregen. De course deed het {R['seeds']} keer.
    """)
    return


@app.cell(hide_code=True)
def verify_cell():
    L.explore("verify-runs", "run", lambda _v: C.verify_frame(int(_v), L.lang), start=C.VERIFY_SHOWN, low=1, high=R["seeds"], step=1, role="",
              task="Loop de runs door. Welke fouten van de eerste versie vangen de controles, en welke missen ze?")
    return


@app.cell(hide_code=True)
def text_verify_after():
    mo.md(C.verify_after(L.lang))
    return


@app.cell(hide_code=True)
def text_source():
    mo.md(rf"""
    ### Experiment 4. Antwoorden uit een bron

    Drie vragen, elk {R['seeds']} keer: *{C.ASK_SATURDAY}* (hoe laat sluit de bibliotheek op zaterdag?) zonder bron; dezelfde vraag met de openingstijden van de bibliotheek als vier genummerde zinnen, en het verzoek om alleen daaruit te antwoorden en de zin te noemen; en *{C.ASK_MAIL}* (kan ik boeken per post terugsturen?), met dezelfde bron, die daar niets over zegt.
    """)
    return


@app.cell(hide_code=True)
def source_cell():
    L.explore("source-runs", "run", lambda _v: C.source_frame(int(_v), L.lang), start=C.SOURCE_SHOWN, low=1, high=R["seeds"], step=1, role="",
              task="Loop de runs door. Waar komt het antwoord vandaan bij elk van de drie, en welke kun je nakijken?")
    return


@app.cell(hide_code=True)
def text_source_after():
    mo.md(C.source_after(L.lang))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om antwoorden te krijgen die je kunt nakijken:

    1. Deel een taak met meerdere stappen op in prompts, één stap per prompt, en geef elk antwoord door aan de volgende prompt.
    2. Kijk tussen twee stappen naar elk antwoord, of laat je programma het nakijken, voordat het verdergaat.
    3. Stel over een bewering die moet kloppen een eigen vraag, in een nieuwe prompt; laat het model het antwoord uitrekenen, en lees zijn stappen, niet alleen zijn ja.
    4. Is er een tekst met het antwoord erin, geef die dan tussen scheidingstekens, en vraag het model alleen daaruit te antwoorden.
    5. Vraag het de zin van elk feit te noemen, en het te zeggen als de tekst geen antwoord op de vraag geeft.
    6. Kijk de citaties na: staat dat in die zin?
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een café wil een korte wekelijkse nieuwsbrief met de specials van deze week. De kok zet ze in een lange notitie aan het personeel, met prijzen, voorraad en wie wanneer werkt. Eén prompt, *Schrijf een nieuwsbrief uit deze notitie*, pikt misschien een verkeerde prijs op, of een gerecht dat niet meer op de kaart staat. Het recept, stap voor stap:

    1. Twee prompts. Prompt 1: *Zet de specials uit de notitie tussen de `<notitie>`-tags op een rij, elk met zijn prijs, één per regel.* Prompt 2: de nieuwsbrief, uit die lijst.
    2. Tussen de twee leest het café de lijst: vijf gerechten, vijf prijzen. Hier ontbreekt één gerecht, dus het café verbetert de lijst vóór prompt 2.
    3. Een prijs moet kloppen, dus voor elk gerecht een nieuwe vraag: *Wat kost de erwtensoep volgens de notitie?*, met de notitie erbij. Waar het antwoord en de lijst verschillen, kijkt het café in de notitie.
    4. Prompt 2 krijgt de lijst tussen `<specials>`-tags: *Schrijf de nieuwsbrief met alleen de gerechten en prijzen tussen de tags.*
    5. Prompt 1 vraagt ook naar de regel van de notitie waar elk gerecht vandaan komt, en *Staat er bij een gerecht geen prijs, schrijf dan: geen prijs.*
    6. Het café kijkt willekeurig twee regels na: staat dat in die regel van de notitie?

    Geen enkele stap heeft een beter model nodig. Elke stap geeft het café een plek om een fout te zien voordat een klant hem ziet.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Welke gewoonte is het?

        Zes manieren waarop mensen een model gebruiken. Deel ze in: welke van de vier is elk?
        """),
        L.task("Zet elk van de zes in zijn groep."),
    ])
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_habits = L.sort("habits")
    sort_habits
    return (sort_habits,)


@app.cell(hide_code=True)
def check_sort(sort_habits):
    result_habits = L.check("habits", sort_habits)
    result_habits
    return (result_habits,)


@app.cell(hide_code=True)
def hints_sort():
    L.hints(
        "habits",
        [
            "Vraag bij elk: zijn er meerdere prompts die elk één stap doen, beoordeelt het model zijn eigen antwoord, wordt een bewering los nagevraagd, of antwoordt het model uit een tekst die het kreeg?",
            "Twee van de zes geven het antwoord van de ene prompt door aan de volgende, en twee laten het model kijken naar wat het zelf schreef. Welke vraagt elke bewering los na, in een nieuwe chat?",
            "De opzet en de datums in de agenda zijn ketens; die met de handleiding is grounding; de jaartallen die één voor één worden nagevraagd, zijn Chain-of-Verification.",
        ],
        why="Eerst de opzet en dan het praatje, en eerst de datums en dan de agenda, zijn prompt chaining: elke prompt doet één stap. Het antwoord nog eens lezen en de eigen samenvatting beoordelen zijn Self-Refine: het model kijkt naar wat het schreef. Elk jaartal los navragen, in een nieuwe chat, is Chain-of-Verification. En antwoorden uit de handleiding, met de paragraaf erbij, is grounding met een citatie.",
    )
    return


@app.cell(hide_code=True)
def text_cite():
    mo.md(r"""
    ### Oefening 2. Welk antwoord kun je nakijken?
    """)
    return


@app.cell(hide_code=True)
def choice_cite_cell():
    choice_cite = L.choice(
        "cite",
        question="Een model kreeg de openingstijden van een bibliotheek als vier genummerde zinnen, en de vraag *Hoe laat sluit de bibliotheek op zaterdag?* Welk antwoord kun je het snelst nakijken?",
        options={
            "sure": "*Om 14.00 uur. Dat weet ik zeker.*",
            "cited": "*Om 14.00 uur [2].*",
            "usual": "*Bibliotheken gaan op zaterdag vaak vroeg dicht, rond 14.00 uur.*",
        },
        correct="cited",
        why={
            "sure": "*Dat weet ik zeker* is nog meer tekst die het model schreef, geen plek waar je kunt kijken. Kijk nog eens naar het nieuwe woord *citatie*.",
            "cited": "[2] zegt waar je moet kijken: je leest zin 2, en ziet of daar 14.00 uur staat.",
            "usual": "Dit antwoord zegt wat bibliotheken vaak doen, niet wat deze bron zegt: er is niets om op te zoeken. Kijk nog eens naar het nieuwe woord *grounding*.",
        },
    )
    choice_cite
    return (choice_cite,)


@app.cell(hide_code=True)
def text_fresh():
    mo.md(r"""
    ### Oefening 3. Waarom een nieuwe prompt?
    """)
    return


@app.cell(hide_code=True)
def choice_fresh_cell():
    choice_fresh = L.choice(
        "fresh",
        question="Waarom krijgt bij Chain-of-Verification elke controle een nieuwe prompt, zonder de eerste versie?",
        options={
            "short": "Korte prompts kosten minder",
            "rule": "Een model kan maar één vraag per prompt beantwoorden",
            "agree": "Met zijn eerste versie voor zich is een model het snel eens met wat het schreef",
        },
        correct="agree",
        why={
            "short": "Korte prompts kosten wel minder, maar daarom is het niet. Wat doet een model met zijn eigen eerste versie voor zich? Kijk nog eens naar het idee van Chain-of-Verification.",
            "rule": "Een model kan meerdere vragen in één prompt beantwoorden. Kijk nog eens naar het idee van Chain-of-Verification: wat moet de controle niet zien?",
            "agree": "Elk token dat een model schrijft, hangt af van alle tekst ervoor. Met de eerste versie in de prompt is het antwoord daaruit het waarschijnlijke; een nieuwe vraag moet op zichzelf beantwoord worden.",
        },
    )
    choice_fresh
    return (choice_fresh,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Vraag het Qwen3 zelf

    In het vak staat de prompt van experiment 4, met de bron. Verander de vraag aan het eind, of de bron, en kijk wat Qwen3 antwoordt, en welke zin het noemt. Qwen3 schrijft het best Engels, dus houd de prompt in het Engels. Het antwoord wordt gesampled met seed 0. Qwen3 draait op je eigen computer, dus niets wat je typt, verlaat hem.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B (1,5 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    C.download_or_stop(get_button, {
        "ask": "*Het chat model Qwen3-0.6B staat nog niet op je computer. De knop downloadt het één keer van Hugging Face, `Qwen/Qwen3-0.6B` in een vaste versie, 1,5 GB, onder de Apache-2.0-licentie, met de tokenizer van Qwen3-0.6B-Base.*",
        "busy": "**Downloaden** {done} van {total} MB …",
        "no_net": "*Het downloaden lukte niet: Hugging Face was niet te bereiken, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*Het downloaden lukte niet: het bestand dat binnenkwam, is niet het vastgezette bestand van de course, dus de course heeft het weggegooid. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*Het downloaden lukte niet: er is geen ruimte op je schijf voor 1,5 GB. Maak ruimte, en klik opnieuw.*",
        "failed": "*Het downloaden lukte niet ({why}). Klik opnieuw.*",
    })
    mo.md("*Qwen3-0.6B is gedownload.*")
    return


@app.cell(hide_code=True)
def prompt_box_cell():
    prompt_box = mo.ui.text_area(value=C.grounded(C.ASK_SATURDAY)[0]["content"], label="Je prompt", rows=9, full_width=True)
    ask_button = mo.ui.run_button(label="Vraag het Qwen3")
    mo.vstack([prompt_box, ask_button])
    return ask_button, prompt_box


@app.cell(hide_code=True)
def demo_answer(ask_button, prompt_box):
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif not ask_button.value:
        _out = mo.md("*Druk op de knop om het Qwen3 te vragen.*")
    elif not prompt_box.value.strip():
        _out = mo.md("*Schrijf eerst een prompt in het vak.*")
    else:
        _words = {"title": "Wat Qwen3 antwoordt", "loading": "Qwen3 laden …", "writing": "Qwen3 schrijft …",
                  "done": "{tokens} tokens in {seconds} seconden; de zinnen die het noemt: {cited}", "nothing": "geen",
                  "load": "Het laden van Qwen3 stopt met een fout. Sluit andere programma's en druk opnieuw op de knop; gebeurt het weer, meld het dan op GitHub.",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.ask_live(prompt_box.value, lambda _html: mo.output.replace(mo.Html(_html)), _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl) veranderde de vorm van één prompt het antwoord; hier geven prompts na elkaar je plekken om de antwoorden na te kijken. Later in de course, in *Wat een agent is: een model, tools en een loop*, vraagt een programma een model steeds opnieuw, zoals in een keten, maar kiest het model zelf de volgende stap, en roept het tools aan. In het domein *Bouwen met AI* komt grounding terug: een programma zoekt eerst de juiste tekst, en dan antwoordt het model daaruit.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_cite, choice_fresh, result_habits):
    L.wrap_up(result_habits, choice_cite, choice_fresh)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
