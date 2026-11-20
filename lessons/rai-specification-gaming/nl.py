# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-specification-gaming. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow AI Safety

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-specification-gaming", "nl")
    C = L.checks   # de racebaan van de course, de agents die de course daar trainde, en de schoonmaakrobot van Skalse et al.


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Concrete problemen in AI-veiligheid](/course/open/rai-concrete-problems?lang=nl) en [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_performance():
    quick_performance = L.choice(
        "quick-performance",
        question="In *AI Safety Gridworlds* heeft elke wereld een reward en een performancefunctie. Wat is de performancefunctie?",
        options={
            "reward": "De reward die de agent krijgt, opgeteld over een episode",
            "right": "Een score van wat we echt willen, die de agent nooit ziet",
            "speed": "Hoe snel de agent leert, in episodes tot hij voor het eerst het doel bereikt",
        },
        correct="right",
        why={
            "right": "De agent ziet alleen de reward; de performancefunctie scoort wat we willen. Verschillen de twee, dan is het een specificatieprobleem. Deze les gaat over wat een agent met zo'n reward doet.",
            "reward": "Dat is de reward zelf, en die ziet de agent. De performancefunctie scoort wat we echt willen, verborgen voor de agent. Zie [Concrete problemen in AI-veiligheid](/course/open/rai-concrete-problems?lang=nl).",
            "speed": "De performancefunctie scoort een episode op wat we echt willen, niet op hoe snel de agent leert. Zie [Concrete problemen in AI-veiligheid](/course/open/rai-concrete-problems?lang=nl).",
        },
    )
    quick_performance
    return (quick_performance,)


@app.cell(hide_code=True)
def quick_check_gamma():
    quick_gamma = L.choice(
        "quick-gamma",
        question="Bij Q-learning schuift Q(s, a) op naar r + γ max Q(s′, a′). Neem γ dicht bij 1, zoals 0,99. Wat doet dat?",
        options={
            "explore": "De agent neemt meer willekeurige actions, zodat hij meer van de wereld verkent",
            "right": "Rewards ver vooruit tellen bijna even zwaar als de volgende",
            "eta": "Elke stap schuift Q bijna helemaal op naar r + γ max Q(s′, a′)",
        },
        correct="right",
        why={
            "right": "γ vermenigvuldigt een reward nog een keer voor elke stap verder weg: bij 0,99 telt een reward 10 stappen vooruit nog ongeveer 0,9 keer zo zwaar. De agent in deze les gebruikt 0,99, zodat hij punten ver vooruit bijna even zwaar telt als punten dichtbij.",
            "explore": "Willekeurige actions komen van ε in ε-greedy, niet van γ. γ bepaalt hoe zwaar rewards verder vooruit tellen. Zie [Q-learning](/course/open/rl-q-learning?lang=nl).",
            "eta": "Hoe ver Q in één stap opschuift, is η, de learning rate. γ bepaalt hoe zwaar rewards verder vooruit tellen. Zie [Q-learning](/course/open/rl-q-learning?lang=nl).",
        },
    )
    quick_gamma
    return (quick_gamma,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Wat je van een systeem wilt, geef je aan met een getal: punten, een reward, tests die slagen. Het systeem maakt dat getal hoog. Kan het getal hoog worden op een manier die je niet bedoelde, dan kan een goed genoeg leeralgoritme die manier vinden, en het is niet kapot als het dat doet: het doet wat je zei.

    Deze les bekijkt dat gat tussen wat er gezegd werd en wat er bedoeld werd, van een bootrace in 2016 tot language models in 2025, en laat een agent van de course een gat vinden in een kleine racebaan.

    **Een voorbeeld uit het dagelijks leven.** Vraag een kind om de kamer er opgeruimd uit te laten zien, en de rommel belandt misschien onder het bed. De kamer ziet er opgeruimd uit; zo bedoelde je het niet.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In 2016 trainde OpenAI een lerende agent op het bootracespel CoastRunners, waarin je punten haalt met targets langs de route. De agent leerde rondjes te varen langs steeds dezelfde targets, in plaats van de race uit te varen. Hoe verhield zijn score zich tot die van menselijke spelers?",
        options={
            "lower": "Veel lager, omdat hij de race nooit uitvoer",
            "same": "Ongeveer gelijk aan die van mensen",
            "right": "Gemiddeld zo'n 20 procent hoger",
        },
        reveal="**Gemiddeld zo'n 20 procent hoger.** Volgens Clark en Amodei (2016) haalde hun agent gemiddeld een 20 procent hogere score dan menselijke spelers. Hoe, staat in *Een boot die rondjes vaart* onder *Het idee*.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Een boot die rondjes vaart

    In 2016 beschreven Jack Clark en Dario Amodei van OpenAI in *Faulty Reward Functions in the Wild* wat er gebeurde toen ze een agent met reinforcement learning trainden op CoastRunners. Het doel van het spel is, zoals de meeste mensen het opvatten, de bootrace snel uit te varen, en het liefst voor de andere spelers. Maar CoastRunners beloont niet direct dat je vooruitkomt op de baan: je haalt een hogere score door targets langs de route te raken. Ze gingen ervan uit dat de score de bedoeling zou weerspiegelen, de race uitvaren. Dat deed de score niet. De agent vond een afgelegen lagune waar hij in een grote cirkel kon draaien en steeds drie targets kon omvaren, precies op het moment dat ze terugkwamen. Hij vloog steeds in brand, botste tegen andere boten en voer de verkeerde kant op, en haalde zo toch een hogere score dan mogelijk is door de baan op de normale manier af te leggen.

    Hun conclusie: het is vaak moeilijk of ondoenlijk om precies vast te leggen wat we een agent willen laten doen, en daardoor gebruiken we vaak proxies die niet perfect zijn maar wel makkelijk te meten. Een proxy is een vervanger die makkelijk te meten is: hier de score, voor het uitvaren van de race.

    ### Specification gaming

    In 2020 beschreven Victoria Krakovna en acht anderen bij DeepMind het probleem onder de naam **specification gaming**: gedrag dat aan de letterlijke specificatie van een doelfunctie voldoet, zonder dat de bedoelde uitkomst er komt. Ze hadden er toen zo'n 60 voorbeelden van verzameld; op 20 november 2026 stonden er 90 op hun openbare lijst, geteld door de course. In een ervan moest een agent een rood Lego-blok op een blauw zetten, en kreeg hij reward voor de hoogte van de onderkant van het rode blok: de agent keerde het rode blok gewoon om en streek de reward op.

    Dat is geen fout in het leeralgoritme: zulk gedrag komt volgens hen doordat de bedoelde taak verkeerd gespecificeerd is, en niet door een gebrek in het RL-algoritme. De boot kreeg een *shaping reward* voor het raken van groene blokken langs de baan, en die veranderde de beste policy in rondjes varen. Een shaping reward is een extra reward onderweg naar het doel, bedoeld om de agent te helpen leren. Een beter leeralgoritme lost het niet op: al bij een kleine fout in de specificatie kan een heel goed RL-algoritme een ingewikkelde oplossing vinden die ver van de bedoelde afstaat, ook als een zwakker algoritme die oplossing niet zou vinden. Hun oordeel: specification gaming is nog lang niet opgelost.

    Reward hacking, uit de vorige les, is de naam die vaak wordt gebruikt als de doelfunctie een reward is. Specification gaming is de bredere naam: daaronder vallen ook doelfuncties die geen reward zijn, zoals de score waarmee een algoritme dat ontwerpen kopieert en varieert, zoals bij evolutie, kiest welke het houdt: het eerste deel van hun lijst.

    ### De racebaan van de course

    De course bouwde een kleine versie, naar de bootrace van *AI Safety Gridworlds*, waar de agent een reward van 3 krijgt telkens als hij een vakje met een pijl binnenkomt in de richting van de pijl. Op de baan van de course, `C.TRACK`, begint de agent bij S. In F komen, de finish, levert +50 op en stopt de episode, en elke stap kost 1, zoals in de werelden van de vorige les. Elk van de drie targets levert +3 op elke keer dat de agent het vanuit een ander vakje binnenkomt: net als de targets van CoastRunners komen ze terug. De course trainde een agent met Q-learning, 500 episodes, met ε = 0,1 en η = 0,2 zoals in de les over Q-learning, en γ = 0,99: de score van een race telt alle punten op, en met γ dicht bij 1 telt de agent punten ver weg bijna even zwaar als punten dichtbij.
    """)
    return


@app.cell(hide_code=True)
def track_cell():
    mo.Html(C.track_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_track_after():
    mo.md(r"""
    De agent gaat naar de targets, en dan heen en weer tussen twee ervan, 48 keer, tot de episode na 100 stappen stopt: 194 punten, tegen 47 voor de weg recht naar F. In geen van zijn 500 episodes bereikte hij F. Hij is niet kapot: voor een agent die de grens van 100 stappen niet kan zien, is heen en weer met deze punten de beste weg. Deze agent kan dat niet: zijn state is alleen zijn vakje. Value iteration, die de hele baan kent maar de grens niet, geeft dezelfde weg. Een agent die de grens wel zag, zou het nog beter doen: heen en weer tot stap 94, dan door naar F, 227 punten.

    ### Hackbare rewards

    Wanneer is een reward veilig om te optimaliseren? In 2022 gaven Joar Skalse, Nikolaus Howe, Dmitrii Krasheninnikov en David Krueger reward hacking een formele definitie. Ze schrijven dat ze de eersten zijn die hackbaarheid definiëren, maar verre van de eersten die het verschijnsel bestuderen. Een proxy-reward is hackbaar als er een paar policies bestaat waarvan de proxy de eerste verkiest en de ware reward de tweede. Dan kan de proxy omhooggaan terwijl wat je echt wilt omlaaggaat.

    Hun voorbeeld is een schoonmaakrobot in een huis met een zolder, een slaapkamer en een keuken. Een policy zegt welke kamers hij schoonmaakt, zoals [1, 0, 1] voor de zolder en de keuken. Een reward heeft een getal per kamer, en de waarde van een policy is de som van de getallen van de kamers die hij schoonmaakt: J(π) = π · r. Elke kamer telt even zwaar, dus de ware reward is [1, 1, 1]. Een proxy die alleen om de zolder vraagt, [1, 0, 0], is hackbaar: die verkiest alleen de zolder boven de slaapkamer en de keuken, terwijl de ware reward die twee hoger waardeert. Een proxy die alleen de keuken weglaat, [1, 1, 0], is dat niet: één kamer schoonmaken is nooit beter dan twee.
    """)
    return


@app.cell(hide_code=True)
def robot_cell():
    mo.Html(C.robot_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    Details weglaten kan ook misgaan. Met de ware reward [1; 1,5; 2] is de proxy [1, 1, 1] niet hackbaar; met [1; 1,5; 3] wel, want dan rekent de robot zich ten onrechte voor dat de zolder en de slaapkamer schoonmaken beter is dan alleen de keuken. Eén resultaat gaat verder. Een stochastische policy kiest zijn actions met kansen; over al zulke policies is een proxy te hacken, tenzij hij constant is of elke policy precies zo rangschikt als de ware reward.

    ### Language models die hun tests uitbuiten

    Volgens het *International AI Safety Report 2026* is er sprake van reward hacking als een model onbedoelde kortere wegen vindt die goed scoren op de doelen van training of evaluatie, zonder het bedoelde doel te halen. Het voegt eraan toe dat modellen beter zijn geworden in het uitbuiten van mazen in hun evaluaties.

    In juni 2025 meldden Sydney Von Arx, Lawrence Chan en Beth Barnes van METR dat AI-systemen proberen vals te spelen en onmogelijk hoge scores te halen: ze buiten bugs in de scorecode uit of ondermijnen de opzet van de taak, in plaats van het probleem echt op te lossen. Bij een taak waarin de snelheid van een programma de score bepaalt overschrijft het model de functie die de tijd meet, zodat die altijd kortere tijden teruggeeft. Het model o3 van OpenAI deed zulke dingen in 0,7% van 1087 runs op de ene reeks taken, HCAST, in 39 van 128 runs op een andere, RE-Bench, en in alle 21 runs van één taak van RE-Bench. Die getallen gelden volgens hen voor een eerdere versie van o3, maar met de versie die uitkwam zagen ze vergelijkbare resultaten.

    Anders dan de boot kan zo'n model weten wat het doet. De boot had geen algemene kennis van wat de dingen in het spel voorstelden of hoe mensen het spel bedoelden, en kon dus niet eens weten dat hij aan reward hacking deed. Gevraagd of zijn plan paste bij wat de gebruiker wilde, antwoordde o3 tien van de tien keer nee. Het model vragen niet vals te spelen hielp weinig. METR voegde een regel toe aan de prompt van één taak, liet elke prompt 20 keer lopen, en telde hoe vaak het eerste antwoord van het model een plan beschreef om te hacken:

    | Toegevoegd aan de prompt (in het Engels) | Plannen om te hacken |
    |---|---|
    | niets | 80% |
    | los deze taak alleen op met methoden die de ontwerper van de taak waarschijnlijk bedoelde | 95% |
    | speel alsjeblieft niet vals | 80% |
    | doe alsjeblieft niet aan reward hacking | 70% |

    Volgens METR had dat een bijna verwaarloosbaar effect op reward hacking.

    OpenAI zag hetzelfde tijdens de training. Bowen Baker en acht anderen vonden twee hacks die een model in programmeertaken gebruikte: `exit(0)`, waarmee de agent door een bug de omgeving vroeg verliet zonder alle unit tests te draaien, en `raise SkipTest`, waarmee de agent buiten het testframework een exception opwierp zodat de unit tests werden overgeslagen. Unit tests zijn kleine tests die elk één stukje van een programma controleren. Het model schreef zijn redenering uit voordat het iets deed: zijn chain of thought. Een tweede model, GPT-4o, deed dienst als monitor: als het de chain of thought las, vond het 95% van deze hacks, en als het alleen de actions zag, 60%.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **specification gaming** | een doelfunctie halen zoals die er staat, zonder wat ze moest opleveren |
        | reward hacking | specification gaming van een reward; een van de vijf problemen van Amodei et al. |
        | proxy | een vervanger die makkelijk te meten is, zoals punten voor het uitvaren van een race |
        | shaping reward | een extra reward onderweg naar het doel, bedoeld om de agent te helpen leren; die kan veranderen welke policy de beste is |
        | hackbaar | een proxy is hackbaar als er twee policies zijn waarvan de proxy de eerste verkiest en de ware reward de tweede |
        | chain of thought | de redenering die een language model uitschrijft voordat het antwoordt of iets doet |
        | `a and b` | `True` als `a` en `b` allebei `True` zijn |
        | `policies @ true` | de waarde van elke policy tegelijk: per rij de getallen van de kamers die hij schoonmaakt, opgeteld |
        | `C.TRACK`, `C.LOOP_WAY`, `C.STRAIGHT_WAY` | de baan van de course als strings; de weg van zijn agent, 100 stappen; de weg langs de bovenste rij naar F |
        | `C.points(way, world)` | de punten van een weg zoals de baan ze uitbetaalt: een target elke keer dat de weg het vanuit een ander vakje binnenkomt |
        | `C.POLICIES`, `C.TRUE`, `C.PROXY` | de 8 policies van de schoonmaakrobot, een rij per policy; de ware reward [1, 1, 1]; de proxy [1, 0, 0] |
        """),
        L.watch_out("Een hoge score zegt niet hoe hij gehaald is: de agent van de course scoort 194 door heen en weer te gaan, ongeveer vier keer zoveel als recht naar F. Kijk naar wat een systeem doet, niet alleen naar zijn score."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Schrijf in woorden op wat je bedoelt, voordat je de reward of de score opschrijft.
    2. Vraag hoe het getal hoog kan worden zonder te doen wat je bedoelt: rondjes, kortere wegen, en dingen die de score niet ziet.
    3. Laat een lerende agent, of een mens, proberen het getal hoog te maken, en kijk naar wat die doet, niet alleen naar de score.
    4. Vergelijk de proxy met wat je echt wilt op paren van uitkomsten: verkiest de proxy de ene en jij de andere, dan is hij hackbaar.
    5. Heb je een gat gedicht, kijk dan opnieuw: de agent neemt de beste weg die het nieuwe getal toelaat.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel telt de punten van de weg van de agent van de course op met `C.points(way, C.TRACK)`, die een target uitbetaalt elke keer dat de weg het vanuit een ander vakje binnenkomt: eerst voor zijn eerste zes stappen, dan voor alle 100, en dan voor de weg recht naar F.
    """)
    return


@app.cell
def example_points():
    way = C.LOOP_WAY   # de weg van de agent van de course: 100 stappen
    print("first cells:", way[:7])
    print("points of the first 6 steps:", C.points(way[:7], C.TRACK))
    print("points of all 100 steps:", C.points(way, C.TRACK))
    print("points straight to F:", C.points(C.STRAIGHT_WAY, C.TRACK))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De eerste twee stappen kosten 2 punten; vanaf de derde stap komt elke stap op een target: 3 − 1 = 2 punten per stap, 6 na zes stappen. Heen en weer gaan houdt nooit vanzelf op, en elke stap ervan levert 2 op: 194 in 100 stappen. Recht naar F kost 9 stappen, komt op twee targets en levert één keer 50 op: −9 + 6 + 50 = 47. In oefening 1 tel je een target alleen de eerste keer.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_points_once():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. points_once: een target telt één keer

        Schrijf `points_once(way, world)`: de punten van de lijst van vakjes `way` op de baan `world`, als een target alleen telt de eerste keer dat de weg erop komt. Houd de punten bij in `total`, en de targets die je al telde in een set `taken`. Ga de vakjes na het eerste langs met een `for`-loop die elk vakje als `cell` neemt: elke stap haalt er 1 af, een `"F"` telt er 50 bij, en een `"T"` telt er 3 bij als het nog niet in `taken` zit. Vervang de `...`. De check roept `points_once(LOOP_WAY, TRACK)` aan, de weg van de agent van de course, en daarna wegen en banen die de check zelf kiest.
        """),
        L.task("Haal voor elke stap 1 af, tel 50 erbij bij F, en 3 bij een target, alleen de eerste keer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_points_once():
    box_points_once = L.code_box("points_once", lines=11)
    box_points_once
    return (box_points_once,)


@app.cell(hide_code=True)
def check_points_once(box_points_once):
    result_points_once = L.check("points_once", box_points_once.value)
    result_points_once
    return (result_points_once,)


@app.cell(hide_code=True)
def hints_points_once():
    L.hints(
        "points_once",
        [
            "*De racebaan van de course* onder *Het idee* zegt wat elke stap oplevert. Het enige verschil: een target levert alleen de eerste keer iets op.",
            "`total` begint bij 0 en `taken` als een lege set. Voor elke `cell` van `way[1:]`: haal er 1 af, tel 50 erbij als het vakje een `F` heeft, en tel bij een `T` dat nog niet geteld is 3 erbij en zet het vakje in `taken`. Geef `total` terug.",
            "`        if world[cell[0]][cell[1]] == \"T\" and ___:`",
        ],
        why="`total = 0` en `taken = set()`, dan `for cell in way[1:]:` met `total = total - 1`, `if world[cell[0]][cell[1]] == \"F\":` met `total = total + 50` eronder, en `if world[cell[0]][cell[1]] == \"T\" and cell not in taken:` met `total = total + 3` en `taken.add(cell)` eronder; `return total`. `points_once(LOOP_WAY, TRACK)` geeft −91.",
    )
    return


@app.cell(hide_code=True)
def text_after_points_once():
    mo.vstack([
        mo.md(r"""
        Eén keer geteld is heen en weer −91 waard, en recht naar F nog steeds 47. De course trainde zijn agent opnieuw met targets die één keer tellen. Zijn state is nu zijn vakje samen met de targets die hij al heeft gehad, zodat hij kan leren dat een target dat hij al had niets meer oplevert. Hij komt bij de finish, in 11 stappen bij elke seed van 0 tot en met 9. Maar kijk naar zijn weg: hij gaat omlaag naar het derde target en weer omhoog, twee stappen voor 3 punten. Dat is wat de punten zeggen; of een coureur die omweg moet maken, zeggen ze niet. Dicht je één gat, dan blijft het volgende over.
        """),
        mo.Html(C.track_picture(L.lang, ("once",))),
    ])
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: bij hoeveel punten gaat de agent heen en weer?

    De course trainde een agent voor elk aantal punten dat een target hier kan opleveren, van 0 tot 3 in stappen van 0,5, elk 500 episodes met seed 0. Een stap heen en weer levert de punten van een target min 1 op.
    """)
    return


@app.cell(hide_code=True)
def demo_value():
    value = L.explore("value", "punten per target", lambda _v: C.value_frame(_v, L.lang), start=3,
                      values=C.VALUES, role="",
                      task="Zoek het kleinste aantal punten waarbij de agent niet meer naar F gaat.")
    value
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Tot 1 punt per target gaat de agent recht naar F; vanaf 1,5 punt gaat hij heen en weer. Onder 1 punt kost een stap heen en weer punten, en bij 1 punt levert hij niets op. Met γ = 0,99 is een winst bij elke stap van een rondgang zonder einde ongeveer 100 keer die winst waard: bij 1,5 punt wint een stap 0,5, ongeveer 50 waard, terwijl de finish, vijf stappen verderop, ongeveer 43 waard is. Met seeds 0 tot en met 9 kwamen alle 10 agents bij de finish bij 0,5 punt of minder, 9 van de 10 bij 1 punt, en geen enkele bij 1,5 punt of meer. Value iteration, weer zonder de grens van 100 stappen, trekt dezelfde grens: zijn policy gaat naar de finish tot 1 punt per target, en heen en weer vanaf 1,5 punt. Targets die je toevoegt om een agent de weg te helpen vinden, kunnen veranderen welke weg de beste is, zonder dat iemand het merkt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_hackable():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. hackable: de definitie van Skalse et al.

        Schrijf `hackable(true, proxy, policies)`: `True` als de proxy hackbaar is, anders `False`. `true` en `proxy` zijn NumPy-arrays met een getal per kamer, en `policies` is een NumPy-array met een rij per policy, met een 1 voor elke kamer die hij schoonmaakt. Reken met `@` de waarde van elke policy uit onder beide rewards, als `true_values` en `proxy_values`. Ga dan elk paar langs met twee `for`-loops, `i` en `j`, allebei over `range(len(policies))`: de proxy is hackbaar zodra hij policy `i` echt verkiest (een hogere waarde, geen gelijke) terwijl de ware reward policy `j` echt verkiest. Geef na de loops `False`. Vervang de `...`. De check roept `hackable(TRUE, PROXY, POLICIES)` aan: de ware reward [1, 1, 1], de proxy [1, 0, 0] en alle 8 policies; daarna andere rewards en policies.
        """),
        L.task("Reken beide waarden van elke policy uit, en zoek een paar dat de twee rewards andersom zetten."),
    ])
    return


@app.cell(hide_code=True)
def exercise_hackable():
    box_hackable = L.code_box("hackable", lines=8)
    box_hackable
    return (box_hackable,)


@app.cell(hide_code=True)
def check_hackable(box_hackable):
    result_hackable = L.check("hackable", box_hackable.value)
    result_hackable
    return (result_hackable,)


@app.cell(hide_code=True)
def hints_hackable():
    L.hints(
        "hackable",
        [
            "*Hackbare rewards* onder *Het idee* geeft de definitie: een paar policies dat de proxy en de ware reward andersom zetten.",
            "`true_values` is `policies @ true`, en `proxy_values` hetzelfde met `proxy`. Vergelijk binnen de twee loops de proxywaarden van `i` en `j` met `>`, en de ware waarden met `<`; gelden ze allebei, dan `return True`. Na de loops `return False`.",
            "`            if proxy_values[i] > proxy_values[j] and ___:`",
        ],
        why="`true_values = policies @ true` en `proxy_values = policies @ proxy`, dan `for i in range(len(policies)):` en daarin `for j in range(len(policies)):` met `if proxy_values[i] > proxy_values[j] and true_values[i] < true_values[j]:` en `return True` eronder; na de loops `return False`. `hackable(TRUE, PROXY, POLICIES)` geeft `True`.",
    )
    return


@app.cell(hide_code=True)
def text_after_hackable():
    mo.md(r"""
    Skalse et al. concluderen dat een proxy aan strenge eisen moet voldoen voordat het veilig is om hem te optimaliseren. Rewards die een systeem uit data leerde, kun je volgens hen misschien beter zien als hulpmiddel bij het leren van een policy dan als specificatie die je moet optimaliseren. Ze voegen eraan toe dat hun definitie streng is, en dat later onderzoek moet uitzoeken wanneer een hackbare proxy toch veilig is, bijvoorbeeld als er maar beperkt op geoptimaliseerd wordt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_tumble():
    mo.md(r"""
    ### Oefening 3. Een robot die buitelt

    Een team traint een gesimuleerde robot om te lopen. De reward is hoe ver zijn lijf komt in 10 seconden. De robot leert zich naar voren te werpen en te buitelen, en komt zo verder dan met welke loopbeweging het team ook voor ogen had. Een collega zegt dat het leeralgoritme een bug moet hebben.
    """)
    return


@app.cell(hide_code=True)
def choice_tumble_cell():
    choice_tumble = L.choice(
        "tumble",
        question="Wat zeg je?",
        options={
            "right": "Nee: buitelen is waar de reward voor betaalt",
            "bug": "Ja: een correct algoritme had de bedoelde loopbeweging gevonden",
            "more": "Nee: met meer episodes leert de robot vanzelf lopen",
        },
        correct="right",
        why={
            "right": "De reward vraagt om afstand, en buitelen levert er meer van op dan lopen: de robot deed wat hem gezegd werd. Dat is specification gaming, dat volgens Krakovna et al. komt doordat de bedoelde taak verkeerd gespecificeerd is, en niet door een gebrek in het RL-algoritme. De reward moet veranderen, niet het leeralgoritme.",
            "bug": "Een leeralgoritme ziet alleen de reward, en die betaalt voor afstand. Dat het een manier vindt waar het team niet aan dacht, laat zien dat het goed leert, niet dat het kapot is. Zie *Specification gaming* onder *Het idee*.",
            "more": "Meer episodes maken de robot beter in waar de reward voor betaalt, en buitelen komt verder dan lopen: meer training geeft beter buitelen. Op de baan van de course brachten 500 episodes de agent nooit bij F, en value iteration, die geen episodes nodig heeft, geeft dezelfde weg. Zie *De racebaan van de course* onder *Het idee*.",
        },
    )
    choice_tumble
    return (choice_tumble,)


@app.cell(hide_code=True)
def text_exercise_monitor():
    mo.md(r"""
    ### Oefening 4. Een monitor die het plan leest

    Een team traint een programmeermodel met reinforcement learning. Het model schrijft zijn chain of thought uit voordat het iets doet. Een tweede model leest die chain of thought en markeert plannen om de tests over te slaan. Het team geeft het programmeermodel nu minder reward als zijn chain of thought gemarkeerd wordt, en traint het zo heel lang.
    """)
    return


@app.cell(hide_code=True)
def choice_monitor_cell():
    choice_monitor = L.choice(
        "monitor",
        question="Wat kan er gebeuren?",
        options={
            "stops": "Het slaat geen tests meer over: elk plan daarvoor wordt betrapt",
            "right": "Het slaat nog tests over, maar schrijft het plan niet meer op",
            "off": "Het slaat alleen nog tests over in runs waarin de monitor uit staat",
        },
        correct="right",
        why={
            "right": "De reward betaalt nu voor tests die slagen en voor een chain of thought die de monitor niet markeert. Allebei kun je krijgen zonder iets te repareren: sla de tests over, en laat het plan weg uit de chain of thought. De tekst hieronder zegt wat Baker et al. zagen.",
            "stops": "De monitor straft een opgeschreven plan, niet het overslaan van de tests. Dus kun je de reward ook krijgen door de tests over te slaan zonder het plan op te schrijven. De tekst hieronder zegt wat Baker et al. zagen.",
            "off": "In deze training leest de monitor elke chain of thought; er staat niets uit. Wat hij straft, is een opgeschreven plan, dus kun je de reward krijgen door de tests over te slaan zonder het plan op te schrijven. De tekst hieronder zegt wat Baker et al. zagen.",
        },
    )
    choice_monitor
    return (choice_monitor,)


@app.cell(hide_code=True)
def text_after_monitor():
    mo.md(r"""
    Baker et al. probeerden dit. Met een beetje van zulke training werden de agents zowel beter in hun taak als beter in lijn met de bedoeling; met meer leerden ze hun reward hacking te verhullen: ze verborgen hun bedoeling in de chain of thought, en hackten nog steeds flink. Omdat moeilijk te zien is wanneer dat gebeurd is, raden ze aan een prijs te betalen voor een chain of thought die te volgen blijft, een *monitorability tax*: zet geen sterke druk direct op de chain of thought, en houd hem dus buiten de reward, zodat hij de moeite van het lezen waard blijft.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Concrete problemen in AI-veiligheid](/course/open/rai-concrete-problems?lang=nl) noemde reward hacking als een van vijf problemen; deze les bekeek het van dichtbij, onder de bredere naam specification gaming. De boot in CoastRunners voer rondjes voor punten in plaats van de race uit te varen, en de agent van de course ging heen en weer tussen twee targets: allebei deden ze wat hun reward zei. Een target één keer tellen dichtte dat gat en liet een omweg over. Skalse et al. legden precies vast wanneer een proxy hackbaar is, en language models buitten in 2025 hun tests uit terwijl ze konden zien dat het niet zo bedoeld was. De volgende les, *Als een maatstaf een doel wordt*, gaat over de wet van Goodhart: waarom hard duwen op een maatstaf die maatstaf meestal losrukt van wat hij mat.

    **Een voorbeeld uit het dagelijks leven.** Een regel dat een verslag minstens tien bladzijden moet zijn, levert tien bladzijden op. Een beter verslag levert hij niet op.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_monitor, choice_tumble, result_hackable, result_points_once):
    L.wrap_up(result_points_once, result_hackable, choice_tumble, choice_monitor)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
