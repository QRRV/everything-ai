# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-tsne. Route: course/route/lessen/klassiek-ml.yaml. Checks en plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-tsne", "nl")
    C = L.checks   # de 1.000 testcijfers, hun kaart van PCA en van t-SNE, de kansen op buren van één cijfer, en de plaatjes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [De beste schaduw: PCA](/course/open/ml-pca?lang=nl) en één over softmax, die je classifier uit [Project: een lineaire classifier voor handgeschreven cijfers](/course/open/ml-digits-linear?lang=nl) gebruikte.
    """)
    return


@app.cell(hide_code=True)
def quick_check_pca():
    quick_pca = L.choice(
        "quick-pca",
        question="De eerste twee principal components van 1.000 handgeschreven cijfers hebben samen een explained variance van 0,18. Wat betekent dat?",
        options={
            "right": "De schaduwen van de cijfers op het vlak van die twee componenten bewaren 18% van de spreiding; 82% gaat verloren",
            "digits": "18% van de cijfers komt op de goede plek op dat vlak",
            "labels": "Die twee componenten houden 18% van de cijfers uit elkaar",
        },
        correct="right",
        why={
            "digits": "Explained variance gaat over spreiding, niet over losse cijfers: elk cijfer krijgt een schaduw op het vlak. Zie *Het idee* in [De beste schaduw: PCA](/course/open/ml-pca?lang=nl).",
            "labels": "PCA ziet nooit een label, dus kan het niet zeggen welke cijfers het uit elkaar houdt. Explained variance is het deel van de spreiding dat de schaduwen bewaren. Zie *Het idee* in [De beste schaduw: PCA](/course/open/ml-pca?lang=nl).",
            "right": "Een plaatje van dat vlak laat dus minder dan een vijfde zien van hoe de cijfers verschillen. Vandaag een kaart die iets anders bewaart.",
        },
    )
    quick_pca
    return (quick_pca,)


@app.cell(hide_code=True)
def quick_check_softmax():
    quick_softmax = L.choice(
        "quick-softmax",
        question="Softmax maakt van de getallen 2, 1 en 0 drie kansen. Wat klopt?",
        options={
            "right": "Ze tellen op tot 1, en 2 krijgt de grootste",
            "share": "Ze zijn 2/3, 1/3 en 0: elk getal gedeeld door de som",
            "one": "2 krijgt 1 en de andere twee 0",
        },
        correct="right",
        why={
            "share": "Softmax neemt eerst e tot de macht van elk getal, en deelt die door hun som: 0,665, 0,245 en 0,090. Zo krijgt 0 toch een kans, en werken negatieve getallen ook. Zie [Veel klassen: softmax](/course/open/ml-softmax?lang=nl).",
            "one": "Dat is de argmax, de klasse met het grootste getal. Softmax geeft elk getal een kans: 0,665, 0,245 en 0,090. Zie [Veel klassen: softmax](/course/open/ml-softmax?lang=nl).",
            "right": "0,665, 0,245 en 0,090. Vandaag maakt softmax van afstanden de kans dat het ene voorbeeld het andere als buur kiest.",
        },
    )
    quick_softmax
    return (quick_softmax,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Een cijfer uit QMNIST is 784 getallen, één per pixel: een punt in 784 dimensies. Wil je er 1.000 tegelijk bekijken, dan moeten ze op een platte kaart. PCA zet ze daar neer als hun beste schaduw op een vlak, maar voor deze cijfers bewaart dat vlak maar 17,8% van de spreiding, en je ziet wat er verloren gaat: op de kaart van PCA is het dichtstbijzijnde cijfer bij een cijfer maar voor 38,9% van de cijfers hetzelfde cijfer, tegen 86,4% in de 784 getallen. Groepen die in 784 dimensies los van elkaar liggen, vallen in hun schaduw over elkaar heen.

        **Een voorbeeld uit het dagelijks leven.** Een metrokaart bewaart de afstanden tussen de stations niet, en de bochten van het spoor ook niet. Hij bewaart welk station naast welk ligt, en dat is precies wat je nodig hebt om de weg te vinden.

        De 1.000 testcijfers op het vlak van hun eerste twee principal components. Elke stip is een cijfer, en elk getal staat in het midden van de stippen van dat cijfer.
        """),
        mo.Html(C.picture_map(C.pca_map()["Z"], {"title": "PCA, 2 componenten", "x": "component 1", "y": "component 2",
                                                 "alt": "De 1.000 testcijfers als stippen op het vlak van hun eerste twee principal components, met de getallen 0 tot en met 9 in het midden van hun stippen. De 1'en liggen links en de 0'en rechts; de 2'en, 3'en, 5'en, 6'en en 8'en liggen in het midden over elkaar heen, en de 4'en, 7'en en 9'en daaronder."})),
    ])
    return


@app.cell(hide_code=True)
def predict_same():
    guess_same = L.guess(
        "same-guess",
        question="De methode van vandaag, t-SNE, tekent dezelfde 1.000 cijfers ook op een platte kaart. Voor hoeveel cijfers is op die kaart het dichtstbijzijnde cijfer hetzelfde cijfer?",
        options={
            "pca": "Ongeveer 40%, net als op de kaart van PCA",
            "right": "Ongeveer 85%, bijna net zoveel als in de 784 getallen",
            "all": "100%: t-SNE zet elk cijfer naast zijn eigen soort",
        },
        reveal="**Ongeveer 85%.** Met perplexity 30 is het 85,0%, tegen 86,4% in de 784 getallen en 38,9% op de kaart van PCA. t-SNE zoekt niet de beste schaduw, maar een kaart waarop elk cijfer zijn buren houdt. Veel beter dan de buren die het meekrijgt, wordt het zelden: in de 784 getallen heeft 13,6% van de cijfers een ander cijfer als dichtstbijzijnde buur.",
    )
    guess_same
    return (guess_same,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Houd de buren.** **t-SNE**, kort voor t-distributed stochastic neighbor embedding, tekent voorbeelden met veel features op een platte kaart, en vraagt één ding van die kaart: dat elk voorbeeld zijn buren houdt. Dat doet het in drie stappen.

        1. **Buren in de 784 getallen.** Neem één voorbeeld en zijn afstand $d_j$ tot elk ander voorbeeld $j$. Softmax maakt van de getallen $-d_j^2 / (2 \cdot \text{breedte}^2)$ kansen $p_j$ die optellen tot 1: hoe waarschijnlijk het is dat het voorbeeld $j$ als buur kiest. Voorbeelden dichtbij krijgen er bijna alles van. De breedte, `width` in code, zegt hoe ver het voorbeeld kijkt: een kleine breedte geeft bijna alles aan de paar dichtstbijzijnde, een grote breedte verdeelt het over veel.
        2. **De perplexity legt de breedte vast.** De **perplexity** van de kansen is $e^{-\sum_j p_j \log p_j}$. Voor k gelijke kansen is dat precies k; voor ongelijke is het ongeveer het aantal buren dat telt. Jij kiest één perplexity, zoals 30, en t-SNE zoekt voor elk voorbeeld de breedte die het die perplexity geeft: klein waar voorbeelden dicht op elkaar liggen, groot waar ze ver uit elkaar liggen.
        3. **De kaart.** t-SNE begint met de voorbeelden op een platte kaart, rekent uit hun afstanden op de kaart dezelfde soort kansen uit, en schuift ze met gradient descent tot die dicht bij de kansen in de 784 getallen liggen; de loss komt neer op de cross-entropy tussen die twee. Op de kaart nemen de kansen langzamer af met de afstand dan in stap 1, langs een curve met dikkere staarten die de t-verdeling heet, de t in de naam, zodat groepen ruimte hebben om los van elkaar te liggen.

        Niets in deze stappen bewaart de grootte van een groep, of de afstand tussen twee groepen: alleen wie wiens buur is. Ook language models hebben een perplexity, in een latere les, met dezelfde formule: daar telt die tussen hoeveel woorden een model in feite kiest.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **t-SNE** | een methode die voorbeelden met veel features op een platte kaart tekent, zo dat buren buren blijven | `TSNE(n_components=2, perplexity=30, random_state=0)` |
        | **perplexity** | $e^{-\sum_j p_j \log p_j}$: ongeveer het aantal buren dat voor elk voorbeeld telt; jij kiest hem, t-SNE zoekt de breedtes | `perplexity=30` |
        | `fit_transform(X)` | fit en geeft de kaart in één keer terug; t-SNE heeft geen `transform`, dus een nieuw voorbeeld kan niet op een kaart die al gemaakt is | `TSNE(...).fit_transform(X)` |

        Het plaatje laat de 1.000 testcijfers zien op kaarten van t-SNE, elke stip een cijfer en elk getal in het midden van zijn stippen. Een kaart maken kost scikit-learn even tijd, dus de course heeft ze één keer gemaakt, met `random_state=0`. De assen hebben geen getallen: op een kaart van t-SNE betekenen ze niets. Schuif met de slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_perplexity_cell():
    def _frame(p):
        said = f"Perplexity {p}: voor {L.number(C.same(p) * 100, 1)}% van de cijfers is het dichtstbijzijnde cijfer op de kaart hetzelfde cijfer."
        alt = (f"De 1.000 testcijfers als stippen op de kaart van t-SNE met perplexity {p}, met de getallen 0 tot en met 9 in het midden van hun stippen. "
               + said)
        return (C.picture_map(C.tsne_map(p), {"title": f"t-SNE, perplexity {p}", "x": "kaart 1", "y": "kaart 2", "alt": alt}), said)

    explore_perplexity = L.explore("perplexity", "perplexity", _frame, start=30, values=C.PERPLEXITIES, role="hyper",
                                   task="Ga van perplexity 2 naar 100. Welke twee cijfers liggen op elke kaart naast elkaar?")
    explore_perplexity
    return


@app.cell(hide_code=True)
def text_after_perplexity():
    mo.vstack([
        mo.md(r"""
        Op elke kaart liggen de 9'en het dichtst bij de 4'en. Ook in de 784 getallen raken 4'en en 9'en vaker verward dan elke andere twee cijfers: 17 van de 110 4'en hebben een 9 als dichtstbijzijnde buur, en 6 van de 94 9'en een 4. Voor 84,2% tot 85,4% van de cijfers is het dichtstbijzijnde cijfer op de kaart hetzelfde cijfer, bij elke perplexity. Een andere seed, `random_state=1`, geeft bij perplexity 30 bijna dezelfde kaart, met 86,7%: scikit-learn begint een kaart vanuit PCA, dus de seed verandert weinig.
        """),
        L.watch_out("De grootte van een eiland en de afstand tussen eilanden zeggen weinig. In de 784 getallen liggen de 1'en het dichtst bij elkaar, gemiddeld 4,1 van hun gemiddelde, tegen 5,8 tot 6,8 voor elk ander cijfer; toch zijn bij perplexity 2 en 5 de eilanden van de 0'en, 6'en en 7'en kleiner dan dat van hen. En bij perplexity 30 liggen de 4'en het verst van de 0'en, terwijl dat in de 784 getallen de 1'en zijn."),
        mo.md(r"""
        Wat één cijfer ziet. Het eerste testcijfer is een 7. Het plaatje laat zijn 30 grootste kansen op buren zien, van groot naar klein, voor de breedte op de slider: gevulde staven voor buren die ook een 7 zijn, open staven voor andere cijfers. Schuif met de slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_width_cell():
    def _frame(w):
        f = C.width_facts(w)
        said = (f"Breedte {L.number(w, 1)}: perplexity {L.number(f['perplexity'], 1)}. De dichtstbijzijnde buur krijgt {L.number(f['top'], 3)}, "
                f"{f['over']} buren krijgen meer dan 0,01, en de andere 7'en krijgen samen {L.number(f['same'], 3)}.")
        alt = (f"Staven van de 30 grootste kansen op buren van het eerste testcijfer, een 7, voor breedte {L.number(w, 1)}, op een as "
               "van 0 tot 1, gevuld voor buren die een 7 zijn en open voor andere cijfers. " + said)
        return (C.picture_width(w, {"title": f"Buren van een 7, breedte {L.number(w, 1)}", "x": "buur, dichtstbijzijnde eerst",
                                    "y": "kans", "alt": alt}), said)

    explore_width = L.explore("width", "breedte", _frame, start=1.2, values=C.WIDTHS, role="hyper",
                              task="Ga van breedte 1,2 naar 2,5. Welke breedte geeft een perplexity van ongeveer 30?")
    explore_width
    return


@app.cell(hide_code=True)
def text_after_width():
    mo.md(r"""
    Met een kleine breedte kijkt de 7 bijna alleen naar zijn dichtstbijzijnde buur: bij 1,2 is de perplexity 1,3. Hoe groter de breedte, hoe meer buren de kans delen, en hoe meer ervan naar andere cijfers gaat: bij 2,5 is de perplexity 195,4, en krijgen de andere 7'en nog maar 0,594. Breedte 2,0 geeft 28,6, ongeveer de 30 van de kaarten hierboven. t-SNE vindt zo'n breedte voor elk voorbeeld door te halveren: te veel buren, dan een kleinere breedte; te weinig, dan een grotere.

    ### Recept

    Om voorbeelden met veel features op een platte kaart te bekijken:

    1. Zet de voorbeelden in rijen getallen op één schaal; hier loopt elke pixel van 0 tot 1.
    2. Kies een perplexity, ongeveer het aantal buren dat voor elk voorbeeld telt: 30, de standaard van scikit-learn, is een goed begin.
    3. Maak de kaart: `TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(X)`.
    4. Maak hem opnieuw met een andere perplexity, en vertrouw alleen wat elke kaart laat zien: welke voorbeelden bij elkaar liggen.
    5. Controleer wat je ziet aan de voorbeelden zelf, bijvoorbeeld hoe vaak de dichtstbijzijnde buur hetzelfde label heeft in de oorspronkelijke getallen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De eerste 300 testcijfers. De cel maakt een kaart met PCA en een met t-SNE, en print voor de 784 getallen en voor elke kaart het deel van de cijfers waarvan het dichtstbijzijnde cijfer hetzelfde cijfer is, afgerond op twee decimalen. De opmerkingen met `#` zeggen welke stap van het recept elk deel is. t-SNE duurt een paar seconden. Voer de cel uit.
    """)
    return


@app.cell
def example_tsne():
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # 1. de eerste 300 testcijfers, elk 784 getallen van 0 tot 1, en welk cijfer elk is
    _data = np.load("data/qmnist-small.npz")
    _X = _data["test_images"][:300] / 255
    _y = _data["test_labels"][:300]

    def _same_neighbor(Z):
        # 5. het deel van de cijfers waarvan het dichtstbijzijnde andere cijfer, in Z, hetzelfde cijfer is
        same = 0
        for i in range(len(Z)):
            j = np.argsort(np.linalg.norm(Z - Z[i], axis=1))[1]
            same = same + int(_y[j] == _y[i])
        return same / len(Z)

    # 2. en 3. een kaart met PCA, en een met t-SNE met perplexity 30
    _pca_map = PCA(n_components=2).fit_transform(_X)
    _tsne_map = TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(_X)

    # 5. de 784 getallen, de kaart van PCA en de kaart van t-SNE
    print(round(_same_neighbor(_X), 2), round(_same_neighbor(_pca_map), 2), round(_same_neighbor(_tsne_map), 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `np.argsort` zet de afstanden op volgorde; plek 0 is het cijfer zelf, op afstand 0, dus plek 1 is zijn dichtstbijzijnde andere cijfer. In de 784 getallen is het dichtstbijzijnde cijfer voor 0,82 van deze 300 hetzelfde cijfer; op de kaart van PCA voor 0,4, en op de kaart van t-SNE weer voor 0,82. De 1.000 cijfers van de slider geven hetzelfde beeld: 0,864, 0,389 en 0,85.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_probs():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Kansen op buren

        `neighbor_probs(d, width)` moet de kansen op buren van één voorbeeld teruggeven: `d` bevat zijn afstand tot elk ander voorbeeld. Vervang de twee `...`: het getal voor elke buur, $-d^2 / (2 \cdot \text{breedte}^2)$, en de kansen die softmax van `e` maakt. Elk getal in `z` is 0 of kleiner, dus `np.exp` kan hier niet te groot worden, en je hoeft het grootste getal er niet eerst af te trekken. De check roept `neighbor_probs(d, 1.0)` aan met `d` = [1, 2, 3], en probeert er nog vier: een andere breedte, vier verre buren, vier op dezelfde afstand, en één buur alleen.
        """),
        L.task("Laat `neighbor_probs` van afstanden kansen op buren maken."),
    ])
    return


@app.cell(hide_code=True)
def exercise_probs():
    box_probs = L.code_box("neighbor-probs", lines=7)
    box_probs
    return (box_probs,)


@app.cell(hide_code=True)
def check_probs(box_probs):
    result_probs = L.check("neighbor-probs", box_probs.value)
    result_probs
    return (result_probs,)


@app.cell(hide_code=True)
def hints_probs():
    L.hints(
        "neighbor-probs",
        [
            "Elke buur krijgt min zijn gekwadrateerde afstand gedeeld door 2 keer de breedte in het kwadraat: een grotere afstand geeft een kleiner getal. NumPy rekent het voor heel `d` tegelijk uit.",
            "`d ** 2` kwadrateert elke afstand. De kansen zijn `e` gedeeld door zijn som, zoals bij softmax.",
            "`z = -d ** 2 / (2 * ___ ** 2)` en `return e / np.sum(___)`.",
        ],
        why="Met breedte 1 is `z` −0,5, −2 en −4,5; `e` is 0,607, 0,135 en 0,011, samen 0,753; daardoor gedeeld: 0,806, 0,180 en 0,015.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_perplexity():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Perplexity

        `perplexity(p)` moet de perplexity van de kansen `p` teruggeven. Vervang de twee `...`: `h`, min de som van elke kans keer zijn log, en de perplexity, e tot de macht `h`. De check roept `perplexity(four)` aan met vier kansen van 0,25, en probeert er nog vier: 0,7, 0,2 en 0,1; twee van 0,5; één zekere buur, 0,97, met drie van 0,01; en dertig gelijke.
        """),
        L.task("Laat `perplexity` tellen hoeveel buren er tellen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_perplexity():
    box_perplexity = L.code_box("perplexity", lines=6)
    box_perplexity
    return (box_perplexity,)


@app.cell(hide_code=True)
def check_perplexity(box_perplexity):
    result_perplexity = L.check("perplexity", box_perplexity.value)
    result_perplexity
    return (result_perplexity,)


@app.cell(hide_code=True)
def hints_perplexity():
    L.hints(
        "perplexity",
        [
            "`h` is de cross-entropy van `p` met zichzelf: elke kans keer zijn log, opgeteld, met een min ervoor. De perplexity is e tot de macht `h`.",
            "Werk op de hele array tegelijk: NumPy vermenigvuldigt `p` waarde voor waarde met zijn log, en heeft een functie voor een som en een voor e tot een macht.",
            "`h = -np.sum(p * ___)` en `return np.exp(___)`.",
        ],
        why="Vier van 0,25: de log van 0,25 is −1,386, dus `h` is −4 · 0,25 · (−1,386) = 1,386, en e tot de macht 1,386 is 4.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_islands():
    mo.md(r"""
    ### Oefening 3. De kaart lezen
    """)
    return


@app.cell(hide_code=True)
def choice_islands_cell():
    choice_islands = L.choice(
        "islands",
        question="Op de kaart met perplexity 30 ligt het eiland van de 4'en vlak naast dat van de 9'en, en liggen de 0'en het verst van de 4'en. Wat kun je daaruit afleiden?",
        options={
            "far": "De 0'en en de 4'en zijn de twee cijfers die het meest verschillen",
            "right": "Veel 4'en en 9'en zijn in de 784 getallen elkaars buren; hoe ver de 0'en van de 4'en liggen, belooft de kaart niet",
            "nothing": "Niets: t-SNE zet de eilanden zomaar ergens neer",
        },
        correct="right",
        why={
            "far": "In de 784 getallen ligt het gemiddelde van de 1'en verder van dat van de 0'en, 7,34, dan het gemiddelde van de 4'en, 6,08, en de kaarten met perplexity 2 en 5 zetten de 1'en ook het verst van de 0'en. Hoe ver eilanden uit elkaar liggen, verandert met de perplexity: t-SNE bewaart buren, geen afstanden. Zie de waarschuwing na de eerste slider.",
            "nothing": "t-SNE bewaart buren: voor 85,0% van de cijfers is het dichtstbijzijnde cijfer op de kaart hetzelfde cijfer, bijna de 86,4% van de 784 getallen. Buren op de kaart zijn buren in de data. Zie de eerste slider.",
            "right": "4'en en 9'en raken vaker verward dan elke andere twee cijfers: 23 keer is het ene de dichtstbijzijnde buur van het andere. Dat bewaart t-SNE, dus raken hun eilanden elkaar op elke kaart. De afstand tussen eilanden die geen buren zijn, bewaart het niet.",
        },
    )
    choice_islands
    return (choice_islands,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** t-SNE bewaart waar [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl) op leunt, wie wiens buur is, waar PCA uit [De beste schaduw: PCA](/course/open/ml-pca?lang=nl) de meeste spreiding bewaart. Zijn kansen op buren zijn de softmax uit [Veel klassen: softmax](/course/open/ml-softmax?lang=nl), en zijn perplexity is e tot de macht van een cross-entropy. Eilanden op zijn kaart zijn een reden om beter te kijken, geen clusters: controleer ze met een clustering zoals k-means, en in de data zelf. In de volgende les, *Samen gekocht: support, confidence en lift*, zoek je producten die vaak in dezelfde bestelling belanden.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_islands, result_perplexity, result_probs):
    L.wrap_up(result_probs, result_perplexity, choice_islands)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
