# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-dutch-compounds. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-dutch-compounds", "nl")
    C = L.checks   # de lijst Nederlandse woorden van de course, de tussenklanken en de eigen splitser


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Woorden en hun delen](/course/open/nlp-morphemes?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_split():
    quick_split = L.choice(
        "quick-split",
        question="In *Woorden en hun delen* geeft `split_compound` de eerste knip van links in twee bekende woorden terug, of het woord als één deel. Wat geeft `split_compound(\"voetbalveld\", [\"voet\", \"bal\", \"voetbal\", \"veld\"])`?",
        options={
            "right": "`['voetbal', 'veld']`",
            "three": "`['voet', 'bal', 'veld']`, de drie woorden waaruit het bestaat",
            "whole": "`['voetbalveld']`, omdat dat woord niet in de lijst staat",
        },
        correct="right",
        why={
            "right": "De knippen gaan van links naar rechts. Na *voet* staat de rest, *balveld*, niet in de lijst; na *voetbal* staan beide stukken erin, en daar stopt `split_compound`, met twee stukken. Deze les splitst de stukken opnieuw.",
            "three": "`split_compound` knipt één keer, dus de functie geeft hoogstens twee stukken. Zie oefening 2 in [Woorden en hun delen](/course/open/nlp-morphemes?lang=nl).",
            "whole": "`split_compound` geeft het woord alleen als één deel als geen enkele knip werkt, en hier staan *voetbal* en *veld* allebei in de lijst. Zie oefening 2 in [Woorden en hun delen](/course/open/nlp-morphemes?lang=nl).",
        },
    )
    quick_split
    return (quick_split,)


@app.cell(hide_code=True)
def quick_check_write():
    quick_write = L.choice(
        "quick-write",
        question="Hoe schrijft het Nederlands de samenstelling die het Engels als *science fiction* schrijft?",
        options={
            "right": "Als één woord: *sciencefiction*",
            "space": "Met een spatie ertussen",
            "hyphen": "Met een streepje tussen de twee delen: *science-fiction*",
        },
        correct="right",
        why={
            "right": "Volgens de spellingregels van de Taalunie schrijf je de delen van een samenstelling aan elkaar. Een Nederlands programma krijgt een samenstelling dus als één string, zonder spatie om op te knippen: deze les zoekt waar je knipt.",
            "space": "Het Engels schrijft veel samenstellingen met een spatie, het Nederlands schrijft ze als één woord. Zie *Drie manieren om woorden te maken* in [Woorden en hun delen](/course/open/nlp-morphemes?lang=nl).",
            "hyphen": "Het Nederlands schrijft samenstellingen als één woord, in een paar bijzondere gevallen met een streepje. Zie *Drie manieren om woorden te maken* in [Woorden en hun delen](/course/open/nlp-morphemes?lang=nl).",
        },
    )
    quick_write
    return (quick_write,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Het Nederlands maakt steeds nieuwe samenstellingen, en schrijft elke samenstelling als één woord. Een zoekmachine die bij *fiets* alleen hele woorden vergelijkt, mist een pagina over een *fietsenmaker*. En geen lijst woorden bevat ze allemaal: de naslaggrammatica van het Nederlands, de e-ANS, geeft *kindercarnavalsoptochtvoorbereidingswerkzaamheden* als voorbeeld van een lange samenstelling.

    Programma's die samenstellingen splitsen, doen het beter. Christof Monz en Maarten de Rijke (2002) splitsten de samenstellingen in een verzameling Nederlandse documenten en in de zoekvragen, en voegden de delen toe; gemeten op hun manier ging de kwaliteit van de resultaten van hun zoekmachine met iets meer dan 6 procent omhoog. Philipp Koehn en Kevin Knight (2003) schrijven dat een vertaalprogramma een onbekende samenstelling kan vertalen door de delen te vertalen.

    Deze les splitst Nederlandse samenstellingen in meer dan twee delen, met de klanken die het Nederlands ertussen zet, en op elke manier die de letters toelaten.

    **Een voorbeeld uit het dagelijks leven.** Een lange trein is een rij gekoppelde wagons. Wil je zien wat de trein vervoert, dan zoek je de koppelingen en bekijk je de wagons een voor een.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Monz en De Rijke namen een lijst willekeurig gekozen zelfstandige naamwoorden uit een verzameling Nederlandse teksten om zoekmachines mee te testen, en splitsten elk naamwoord in delen. Hoeveel van de naamwoorden hadden twee of meer delen?",
        options={
            "few": "Ongeveer 5 van de 100",
            "right": "Ongeveer een kwart",
            "most": "Ongeveer driekwart",
        },
        reveal="**Ongeveer een kwart.** In hun tabel hebben 125 van de 535 Nederlandse naamwoorden twee of drie delen: 23,4 van de 100. Ze schrijven dat het om ongeveer 25 procent gaat, ook voor het Duits, en voegen er zelf aan toe dat hun steekproeven veel te klein zijn voor een harde uitspraak. In hun tekst gaat het om 553 Nederlandse naamwoorden, maar de getallen in de tabel tellen op tot 535.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Delen in delen

    *Woorden en hun delen* splitste *voordeur* in twee bekende woorden. Een Nederlandse samenstelling kan meer delen hebben, omdat een deel zelf een samenstelling kan zijn: *voetbalveld* is *voetbal* en *veld*, en *voetbal* is *voet* en *bal*. De e-ANS noemt samenstellen recursief: je kunt blijven samenvoegen, en zo ontstaan de lange voorbeelden daar. **Decompounding**, of compound splitting, is een samenstelling in delen splitsen; Ben Verhoeven en anderen (2014) omschrijven compound splitting als het vinden van de grenzen tussen de delen. Een programma dat *voetbalveld* splitst, moet de delen ook splitsen: hetzelfde werk op een korter woord. Een functie die zichzelf aanroept, doet dat, zoals in [Een functie die zichzelf aanroept](/course/open/py-recursion?lang=nl).

    ### Tussenklanken

    Tussen de twee woorden van een samenstelling hoor je soms een extra klank: *stad* en *muur* worden *stadsmuur*, *hond* en *hok* worden *hondenhok*, en *zon* en *bril* worden *zonnebril*. De e-ANS noemt drie zulke **tussenklanken**: -en-, -e- en -s-; de taaladviessite van de Taalunie, taaladvies.net, noemt er nog een: -er-, zoals in *eiergerecht*. Volgens die site bepalen geen harde regels of een samenstelling een tussenklank krijgt, en kan één woord in verschillende samenstellingen verschillende tussenklanken krijgen: *zonwering*, *zonneschijn*, *zonsverduistering*.

    Voor de spelling van de klank is er wel een regel van de Taalunie: schrijf -en- als het linkerdeel een zelfstandig naamwoord is met een meervoud op -en en zonder meervoud op -es, en anders -e-. Zo is het sinds 1996 *pannenkoek*, omdat *pan* het meervoud *pannen* heeft, en niet meer *pannekoek*. De regel kent uitzonderingen: *zonnebril* houdt -e-, omdat er maar één zon is.

    Het Duits heeft ook tussenklanken. Koehn en Knight lieten ze tussen twee willekeurige woorden toe, omdat er volgens hen geen eenvoudige regels zijn voor wanneer zulke letters erbij komen: *Aktionsplan* (actieplan) is *Aktion*, *s* en *Plan*.

    ### Meer dan één splitsing

    Dezelfde letters kun je vaak op meer manieren splitsen. *valkuil* is *val* en *kuil*, maar ook *valk* en *uil*; volgens de spellingregels van de Taalunie mag je *valk-uil* schrijven als je zo'n vogel bedoelt. Roeland Ordelman, Arjan van Hessen en Franciska de Jong (2003) vonden voor zo'n 2 procent van de samenstellingen die hun programma herkende twee of meer splitsingen. Dat gebeurde bijvoorbeeld bij een samenstelling van drie of meer delen, en dat is meestal geen probleem; maar sommige, zoals *politiekringen*, hebben ook een splitsing die nergens op slaat: *politie-kringen*, en niet *politiek-ringen*. Iemand met Nederlands als moedertaal kan meestal kiezen, schrijven ze, al is daar soms de context voor nodig.

    Koehn en Knight bekijken alle splitsingen in bekende woorden, met het woord zelf erbij, via een zoektocht die recursief alles probeert, en hun bekende woorden hebben minstens drie letters. Voor *Aktionsplan* geeft dat *aktionsplan*, *aktion-plan*, *aktions-plan* en *akt-ion-plan*. Om er een te kiezen, kijkt een van hun methoden hoe vaak elk deel in tekst voorkomt.

    Een splitsing kan ook fout zijn terwijl elk deel een woord is. *kantoor* heeft niets met een *kant* en een *oor* te maken. Het Duitse voorbeeld van Koehn en Knight is *Freitag* (vrijdag), dat splitst in *frei* (vrij) en *Tag* (dag).
    """)
    return


@app.cell(hide_code=True)
def parts_cell():
    mo.Html(C.parts_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **decompounding** | een samenstelling in delen splitsen, zoals *voetbalveld* in *voet*, *bal* en *veld* |
        | **tussenklank** | een klank tussen de twee woorden van een samenstelling, zoals -s- in *stadsmuur* en -en- in *hondenhok* |
        | `C.DUTCH` | de lijst Nederlandse woorden van de course, als set: de 37.123 woorden van Open Dutch WordNet, een vrij beschikbaar Nederlands woordenboek, die één woord van drie of meer kleine letters zijn |
        | `C.LINKS` | de tussenklanken van de course, `["", "s", "en"]`; `""`, de lege string, is geen klank |
        | `x in a_set` | `True` als `x` een van de items van de set is; bij veel items veel sneller dan bij een lijst |
        | `[a] + b` | voor een lijst `b`: een nieuwe lijst met `a` vóór de items van `b`; `+` plakt twee lijsten aan elkaar, zoals twee strings |
        """),
        L.watch_out("De lijst van de course heeft de basisvormen van woorden, zoals *paard* en niet *paarden*, en laat scheldwoorden en grove woorden weg. Het programma van de course laat ook de tussenklank -e- weg, omdat de spelling daarbij meestal verandert: *zon* wordt *zonne* in *zonnebril*. Ook bij -en- kan de spelling veranderen, en dan vindt het programma geen splitsing: *peer* wordt *peren* in *perensap*."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Probeer elke knip van het woord in een linker- en een rechterstuk.
    2. Haal voor elke tussenklank die klank van het eind van het linkerstuk af, en kijk of wat overblijft en het rechterstuk bekende woorden zijn.
    3. Splits de rest van het woord op dezelfde manier, met een functie die zichzelf aanroept; een bekend woord op zichzelf telt ook als splitsing.
    4. Bewaar elke splitsing die je vindt, niet alleen de eerste.
    5. Kijk naar de splitsingen: de letters kunnen passen waar de betekenis niet past.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel haalt elke tussenklank met een slice van het eind van het stuk *honden* af, zoals *Woorden en hun delen* een uitgang afknipte, en print of dat het stuk weer geeft en of wat overblijft in de lijst van de course staat.
    """)
    return


@app.cell
def example_link():
    piece = "honden"
    for sound in ["", "s", "en"]:
        stem = piece[:len(piece) - len(sound)]   # het stuk zonder de klank
        print(f"klank '{sound}':", stem, stem + sound == piece, stem in C.DUTCH)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Alleen -en- past en laat een bekend woord over: *honden* staat niet in de lijst van de course, want die heeft alleen basisvormen, en *hond* wel. Met de lege klank is wat overblijft het hele stuk, dus met dezelfde controle vind je ook een splitsing zonder klank. Oefening 1 doet dit voor elke knip van een woord.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_link():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. link_splits: twee woorden en een tussenklank

        `C.LINKS` is `["", "s", "en"]`. Schrijf `link_splits(word, words, links)`: geef een lijst terug met elke splitsing van `word` in een bekend woord, een tussenklank en een bekend woord, elk als lijst `[stem, link, right]`. Begin met een lege lijst. Probeer de knippen van links naar rechts zoals in *Woorden en hun delen*, met `left = word[:i]` en `right = word[i:]`, en probeer bij elke knip de klanken van `links` in volgorde, met `stem` het linkerstuk zonder de klank, zoals in het uitgewerkte voorbeeld. Eindigt het linkerstuk echt op die klank, en staan `stem` en `right` allebei in `words`, voeg dan `[stem, link, right]` toe aan de lijst. Geef de lijst aan het eind terug, ook als die leeg blijft. Vervang de `...`. De check roept `link_splits("hondenhok", C.DUTCH, C.LINKS)` aan, daarna andere woorden en eigen lijsten, ook een Duits woord.
        """),
        L.task("Probeer elke knip en elke tussenklank, en bewaar elke splitsing in twee bekende woorden."),
    ])
    return


@app.cell(hide_code=True)
def exercise_link():
    box_link = L.code_box("link_splits", lines=10)
    box_link
    return (box_link,)


@app.cell(hide_code=True)
def check_link(box_link):
    result_link = L.check("link_splits", box_link.value)
    result_link
    return (result_link,)


@app.cell(hide_code=True)
def hints_link():
    L.hints(
        "link_splits",
        [
            "Twee loops, de ene in de andere: de buitenste over de knippen, de binnenste over `links`. Doe daarbinnen de stappen van het uitgewerkte voorbeeld.",
            "Wanneer voeg je een splitsing toe aan de lijst: wat moet er gelden voor `stem` en `link` samen, voor `stem` en voor `right`? En waar komt `return`, zodat die na beide loops komt?",
            "`            if ___ and stem in words and right in words:`",
        ],
        why="`found = []`, dan `for i in range(1, len(word)):` met `left = word[:i]` en `right = word[i:]`, en daarbinnen `for link in links:` met `stem = left[:len(left) - len(link)]` en `if stem + link == left and stem in words and right in words:` met daaronder `found.append([stem, link, right])`; na beide loops `return found`. `link_splits(\"hondenhok\", C.DUTCH, C.LINKS)` geeft `[['hond', 'en', 'hok']]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_link():
    mo.md(r"""
    De afbeelding onder de check toont je functie op meer Nederlandse woorden. *huiswerk* heeft twee splitsingen: *huis* en *werk*, en ook *hui* (wei) met -s-. *zonnebril* heeft er geen: met de -e- erbij wordt *zon* daar *zonne*, met een extra *n*. Oefening 2 splitst in meer dan twee delen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_all():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. all_splits: elke manier, met een functie die zichzelf aanroept

        Schrijf `all_splits(word, words)`: geef een lijst terug met elke manier om `word` in bekende woorden te knippen, elk als lijst delen, zonder tussenklanken. Begin met een lege lijst. Staat `word` zelf in `words`, voeg dan `[word]` toe: een bekend woord is een splitsing van één deel. Probeer daarna de knippen van links naar rechts. Staat het linkerstuk `word[:i]` in `words`, laat `all_splits` dan de rest van het woord splitsen, en voeg elke splitsing die je terugkrijgt toe, met het linkerstuk ervoor geplakt met `+`. Geef de lijst aan het eind terug. De rest is elke keer korter, dus de aanroepen stoppen. Vervang de `...`. De check roept `all_splits("voetbalveld", C.DUTCH)` aan, daarna andere woorden en eigen lijsten.
        """),
        L.task("Bewaar het woord zelf als het bekend is, en zet elk bekend linkerstuk vóór elke splitsing van de rest."),
    ])
    return


@app.cell(hide_code=True)
def exercise_all():
    box_all = L.code_box("all_splits", lines=10)
    box_all
    return (box_all,)


@app.cell(hide_code=True)
def check_all(box_all):
    result_all = L.check("all_splits", box_all.value)
    result_all
    return (result_all,)


@app.cell(hide_code=True)
def hints_all():
    L.hints(
        "all_splits",
        [
            "Werk in twee stappen: eerst het woord zelf, dan een `for`-loop over de knippen, zoals in oefening 1. De aanroep voor de rest geeft een lijst splitsingen terug, dus loop die langs met een tweede `for`-loop.",
            "Wat geeft `all_splits(\"veld\", C.DUTCH)` terug, en wat zet je voor *voetbalveld* vóór elk van die splitsingen? Waarom moet `[word]` in de lijst als het woord zelf bekend is?",
            "`            for rest in ___:`",
        ],
        why="`found = []`, dan `if word in words:` met daaronder `found.append([word])`, en `for i in range(1, len(word)):` met `left = word[:i]`, en `if left in words:` met daaronder `for rest in all_splits(word[i:], words):` en `found.append([left] + rest)`; aan het eind `return found`. `all_splits(\"voetbalveld\", C.DUTCH)` geeft `[['voetbalveld'], ['voet', 'bal', 'veld'], ['voetbal', 'veld']]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_all():
    mo.md(r"""
    De afbeelding onder de check toont je functie op meer woorden: *valkuil* en *politiekring* hebben elk twee splitsingen in twee woorden, en *kantoor* heeft er een die alleen bij de letters past. De demo hieronder zet de twee oefeningen samen.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: splits een eigen woord

    Typ een Nederlands woord, en zie elke splitsing die de splitser van de course vindt met `C.DUTCH`: `all_splits`, met de tussenklanken van `link_splits`. Probeer *politiekring*, *zeeroverschatten* of *koningspaarden*: de lijst heeft *paard* en niet *paarden*, dus de enige splitsing is *koning*, -s-, *paar* en *den*.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_word = mo.ui.text(value="verjaardagstaart", label="Een Nederlands woord")
    demo_word
    return (demo_word,)


@app.cell(hide_code=True)
def demo_look(demo_word):
    mo.Html(C.demo_picture(demo_word.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_real():
    mo.md(r"""
    ### Oefening 3. Een echte samenstelling, of alleen de letters?

    Elke splitsing hieronder bestaat uit woorden van de lijst van de course, zoals de functies van deze les ze vinden. Zet elke splitsing in de juiste groep: een echte samenstelling, waarvan de delen de betekenis geven, of een splitsing waarbij alleen de letters passen.
    """)
    return


@app.cell(hide_code=True)
def exercise_real():
    sort_real = L.sort("real")
    sort_real
    return (sort_real,)


@app.cell(hide_code=True)
def check_real(sort_real):
    result_real = L.check("real", sort_real)
    result_real
    return (result_real,)


@app.cell(hide_code=True)
def text_exercise_which():
    mo.md(r"""
    ### Oefening 4. Welke splitsing is bedoeld?

    Naast het woord zelf geeft `all_splits` twee splitsingen van *valkuil*: *val* en *kuil*, en *valk* en *uil*.
    """)
    return


@app.cell(hide_code=True)
def choice_which_cell():
    choice_which = L.choice(
        "which",
        question="Wat bepaalt welke van de twee splitsingen bedoeld is?",
        options={
            "right": "De tekst rond het woord",
            "fewer": "Het aantal delen, want de goede splitsing heeft er minder",
            "longer": "De lengte van het eerste deel: het langste bekende woord aan het begin is het bedoelde",
        },
        correct="right",
        why={
            "right": "Beide splitsingen zijn twee bekende woorden, dus de letters kunnen niet beslissen: de betekenis beslist. In een zin over een val is het een valkuil, in een zin over vogels kan het een soort uil zijn. Ordelman, Van Hessen en De Jong schrijven dat iemand met Nederlands als moedertaal meestal kan kiezen, al is daar soms de context voor nodig.",
            "fewer": "Beide splitsingen hebben twee delen, dus het aantal delen beslist hier niet. Wat verschilt, is de betekenis, en de tekst rond het woord laat die zien. Zie *Meer dan één splitsing* in *Het idee*.",
            "longer": "Met die regel neemt het programma *valk* en *uil*, ook in een tekst over een val. De lengte van een deel zegt niets over de betekenis. Zie *Meer dan één splitsing* in *Het idee*.",
        },
    )
    choice_which
    return (choice_which,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Woorden en hun delen](/course/open/nlp-morphemes?lang=nl) splitste een samenstelling in twee bekende woorden, bij de eerste knip die werkte. Een Nederlandse samenstelling kan meer delen hebben, omdat een deel zelf een samenstelling kan zijn, en tussen twee woorden kan een tussenklank staan, zoals -s- of -en-. Je vond elke splitsing met een tussenklank, en elke splitsing in een willekeurig aantal delen met een functie die zichzelf aanroept, zoals in [Een functie die zichzelf aanroept](/course/open/py-recursion?lang=nl). Samenstellingen splitsen, decompounding, helpt bij zoeken en vertalen. De letters laten vaak meer dan één splitsing toe, en sommige splitsingen passen alleen bij de letters: de betekenis, uit de tekst rond een woord, zegt welke splitsing klopt. De volgende les kijkt naar de soorten woorden waaruit een zin bestaat.

    **Een voorbeeld uit het dagelijks leven.** Een puzzelstukje past soms op twee plekken, en alleen het plaatje erop vertelt je welke plek de goede is.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_which, result_all, result_link, result_real):
    L.wrap_up(result_link, result_all, result_real, choice_which)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
